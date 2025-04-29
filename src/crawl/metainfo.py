import requests
import time
import json
import logging
import os
import re
from bs4 import BeautifulSoup
from typing import Optional, Dict, List, Any
from urllib.parse import urljoin, quote
from dataclasses import dataclass
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle

@dataclass
class PackageData:
    id: str
    name: str
    language: str
    description: Optional[str]
    downloads: int
    stars: int
    created_at: Optional[str]
    updated_at: Optional[str]
    version: Optional[str]
    repository: Optional[str]
    license: Optional[str]
    homepage: Optional[str]
    keywords: List[str]
    authors: List[str]

class BasePackageCrawler:
    def __init__(
        self,
        output_dir: str = "package_data",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
        max_packages: int = 5000,
        github_tokens: Optional[List[str]] = None
    ):
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.max_packages = max_packages
        self.session = requests.Session()
        self.total_collected = 0
        
        # Setup GitHub tokens
        self.github_tokens = github_tokens or os.getenv("GH_TOKENS", "").split(",")
        if not self.github_tokens or not any(self.github_tokens):
            print("No GitHub tokens provided. API rate limits will be restricted.")
            self.github_tokens = [""]  # Use empty token as fallback
        self.token_iterator = cycle(self.github_tokens)  # Create a round-robin iterator for tokens
        
        os.makedirs(output_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def make_request(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict:
        default_headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        if headers:
            default_headers.update(headers)
            
        # Add GitHub token if requesting from GitHub API
        if "api.github.com" in url:
            token = next(self.token_iterator)
            if token:
                default_headers["Authorization"] = f"token {token}"
            
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, headers=default_headers, timeout=self.timeout)
                
                if response.status_code == 200:
                    # First check if URL contains pkg.go.dev - if so, return HTML content
                    if "pkg.go.dev" in url:
                        return {"html_content": response.text}
                    
                    # For other URLs (like GitHub API), try JSON
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        return {"html_content": response.text}
                
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 5
                    self.logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                
            except requests.RequestException as e:
                self.logger.error(f"Request failed: {e}. Attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                return {}

        self.logger.error(f"Failed to fetch from {url} after all retry attempts")
        return {}
    
    def save_package_batch(self, packages: List[PackageData], language: str):
        output_file = os.path.join(self.output_dir, f"{language}_packages.jsonl")
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for package in packages:
                f.write(json.dumps(vars(package), ensure_ascii=False) + '\n')
                
        self.total_collected += len(packages)
        self.logger.info(f"Saved batch of {len(packages)} {language} packages. Total collected: {self.total_collected}")
    
    def crawl_all(self):
        pass

class RustCratesCrawler(BasePackageCrawler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://crates.io/api/v1/crates"
        self.per_page = 90
    
    def fetch_page(self, page_param: Optional[str] = None) -> dict:
        url = self.base_url
        if page_param:
            url = urljoin(url, f"?{page_param}")
        else:
            url = f"{url}?per_page={self.per_page}"
        
        return self.make_request(url)
    
    def process_crate(self, crate: Dict) -> PackageData:
        return PackageData(
            id=crate.get("id", ""),
            name=crate.get("name", ""),
            language="rust",
            description=crate.get("description"),
            downloads=crate.get("downloads", 0),
            stars=crate.get("recent_downloads", 0),  # Using recent_downloads as a proxy
            created_at=crate.get("created_at"),
            updated_at=crate.get("updated_at"),
            version=crate.get("max_version"),
            repository=crate.get("repository"),
            license=crate.get("license"),
            homepage=crate.get("homepage"),
            keywords=crate.get("keywords", []),
            authors=crate.get("authors", [])
        )
    
    def crawl_all(self):
        next_page = None
        
        try:
            while self.total_collected < self.max_packages:
                response = self.fetch_page(next_page)
                crates_data = response.get("crates", [])
                
                if not crates_data:
                    self.logger.warning("No crates data in response")
                    break
                
                packages = [self.process_crate(crate) for crate in crates_data]
                self.save_package_batch(packages, "rust")
                
                meta = response.get("meta", {})
                next_page = meta.get("next_page")
                
                if not next_page:
                    break
                
                time.sleep(0.1)  # Be nice to the server
                
                if self.total_collected >= self.max_packages:
                    break
        
        except Exception as e:
            self.logger.error(f"Error during crawling Rust crates: {e}")
        
        self.logger.info(f"Rust crates crawling completed. Total collected: {self.total_collected}")
        return self.total_collected

class PythonPyPICrawler(BasePackageCrawler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://pypi.org/pypi"
        
    def fetch_package_info(self, package_name: str) -> Dict:
        url = f"{self.base_url}/{package_name}/json"
        return self.make_request(url)
    
    def fetch_top_packages(self) -> List[str]:
        # Get top packages from PyPI Stats JSON
        top_packages = []
        try:
            # Using PyPI Stats JSON that lists popular packages
            response = requests.get("https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.json")
            response.raise_for_status()
            data = response.json()
            for item in data.get("rows", []):
                if len(top_packages) >= self.max_packages:
                    break
                top_packages.append(item.get("project"))
            
            return top_packages
        except Exception as e:
            self.logger.error(f"Error fetching top Python packages: {e}")
            
            # Fallback to a list of known popular packages
            fallback_packages = [
                "numpy", "pandas", "requests", "matplotlib", "django", "flask", 
                "tensorflow", "pytorch", "scipy", "scikit-learn"
            ]
            return fallback_packages[:min(len(fallback_packages), self.max_packages)]
    
    def process_package(self, package_data: Dict) -> PackageData:
        info = package_data.get("info", {})
        stats = package_data.get("urls", [])
        
        download_count = 0
        for url_data in stats:
            download_count += url_data.get("downloads", 0)
        
        # Try to extract repository from project_urls
        repository = None
        project_urls = info.get("project_urls") or {}
        for key, value in project_urls.items():
            if repository is None and any(kw in key.lower() for kw in ["source", "code", "github", "gitlab"]):
                repository = value
                break
        
        return PackageData(
            id=info.get("name", "").lower(),
            name=info.get("name", ""),
            language="python",
            description=info.get("summary"),
            downloads=download_count,
            stars=0,  # PyPI doesn't have star counts
            created_at=info.get("created") or None,
            updated_at=info.get("last_modified") or None,
            version=info.get("version"),
            repository=repository,
            license=info.get("license"),
            homepage=info.get("home_page") or info.get("project_url"),
            keywords=info.get("keywords", "").split() if info.get("keywords") else [],
            authors=[info.get("author")] if info.get("author") else []
        )
    
    def crawl_all(self):
        try:
            top_packages = self.fetch_top_packages()
            self.logger.info(f"Found {len(top_packages)} top Python packages")
            
            packages = []
            for package_name in top_packages:
                if self.total_collected >= self.max_packages:
                    break
                    
                package_data = self.fetch_package_info(package_name)
                if package_data:
                    packages.append(self.process_package(package_data))
                    
                    if len(packages) >= 100:  # Save in batches
                        self.save_package_batch(packages, "python")
                        packages = []
                        
                time.sleep(0.1)  # Be nice to the server
            
            # Save any remaining packages
            if packages:
                self.save_package_batch(packages, "python")
                
        except Exception as e:
            self.logger.error(f"Error during crawling Python packages: {e}")
        
        self.logger.info(f"Python packages crawling completed. Total collected: {self.total_collected}")
        return self.total_collected

class JavaScriptNPMCrawler(BasePackageCrawler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.search_url = "https://registry.npmjs.org/-/v1/search"
        self.package_url = "https://registry.npmjs.org"
    
    def search_packages(self, page: int = 0) -> Dict:
        params = {
            "text": "popularity:>=0.5",
            "size": 100,
            "from": page * 100
        }
        return self.make_request(self.search_url, params)
    
    def fetch_package_info(self, package_name: str) -> Dict:
        url = f"{self.package_url}/{quote(package_name)}"
        return self.make_request(url)
    
    def process_package(self, package_info: Dict, search_data: Optional[Dict] = None) -> PackageData:
        # Extract basic package data
        package_name = search_data.get("name", "")
        
        # Get latest version
        latest_version = ""
        if "dist-tags" in package_info and "latest" in package_info["dist-tags"]:
            latest_version = package_info["dist-tags"]["latest"]
        
        # Get version-specific data
        version_data = {}
        if latest_version and "versions" in package_info and latest_version in package_info["versions"]:
            version_data = package_info["versions"][latest_version]
        
        # Get time data
        created_at = None
        updated_at = None
        if "time" in package_info:
            created_at = package_info["time"].get("created")
            updated_at = package_info["time"].get("modified")
        
        # Get repository URL
        repository = None
        if "repository" in version_data:
            if isinstance(version_data["repository"], dict):
                repository = version_data["repository"].get("url", "")
            else:
                repository = version_data["repository"]
                
        # Clean up GitHub repository URLs
        if repository and "github.com" in repository:
            repository = repository.replace("git+", "").replace("git:", "https:").replace(".git", "")
        
        # Get keywords
        keywords = []
        if "keywords" in version_data and version_data["keywords"]:
            if isinstance(version_data["keywords"], list):
                keywords = version_data["keywords"]
            elif isinstance(version_data["keywords"], str):
                keywords = version_data["keywords"].split(",")
        
        # Get authors
        authors = []
        if "author" in version_data:
            if isinstance(version_data["author"], dict) and "name" in version_data["author"]:
                authors.append(version_data["author"]["name"])
            elif isinstance(version_data["author"], str):
                authors.append(version_data["author"])
        
        return PackageData(
            id=package_name,
            name=package_name,
            language="javascript",
            description=version_data.get("description"),
            downloads=search_data.get("score", {}).get("detail", {}).get("popularity", 0) * 10000,
            stars=search_data.get("score", {}).get("detail", {}).get("quality", 0) * 100,
            created_at=created_at,
            updated_at=updated_at,
            version=latest_version,
            repository=repository,
            license=version_data.get("license"),
            homepage=version_data.get("homepage"),
            keywords=keywords,
            authors=authors
        )
    
    def crawl_all(self):
        try:
            page = 0
            
            packages = []
            while self.total_collected < self.max_packages:
                search_results = self.search_packages(page=page)
                search_objects = search_results.get("objects", [])
                
                if not search_objects:
                    break
                
                for obj in search_objects:
                    if self.total_collected >= self.max_packages:
                        break
                        
                    package_data = obj.get("package", {})
                    package_name = package_data.get("name")
                    
                    if package_name:
                        package_info = self.fetch_package_info(package_name)
                        if package_info:
                            packages.append(self.process_package(package_info, package_data))
                            
                            if len(packages) >= 100:  # Save in batches
                                self.save_package_batch(packages, "javascript")
                                packages = []
                    
                    time.sleep(0.1)  # Be nice to the server
                
                page += 1
            
            # Save any remaining packages
            if packages:
                self.save_package_batch(packages, "javascript")
                
        except Exception as e:
            self.logger.error(f"Error during crawling JavaScript packages: {e}")
        
        self.logger.info(f"JavaScript packages crawling completed. Total collected: {self.total_collected}")
        return self.total_collected

class GoPackageCrawler(BasePackageCrawler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.search_url = "https://pkg.go.dev/search"
        self.pkg_url = "https://pkg.go.dev"
    
    def fetch_popular_packages(self) -> List[str]:
        popular_packages = []
        seen_repos = set()  # Track unique repositories
        page = 1
        
        try:
            # Use GitHub API to get popular Go repositories
            while len(popular_packages) < self.max_packages:
                github_search_url = "https://api.github.com/search/repositories"
                params = {
                    "q": "language:go stars:>100",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 100,
                    "page": page
                }
                
                # Add GitHub token for authentication
                token = next(self.token_iterator)
                headers = {"Accept": "application/vnd.github.v3+json"}
                if token:
                    headers["Authorization"] = f"token {token}"
                
                self.logger.info(f"Fetching page {page} of GitHub Go repositories...")
                response = self.make_request(github_search_url, params=params, headers=headers)
                
                if not response or not isinstance(response, dict) or "items" not in response:
                    self.logger.error(f"Invalid response from GitHub API: {response}")
                    break
                
                items = response.get("items", [])
                if not items:  # No more results
                    self.logger.info("No more repositories found.")
                    break
                
                for item in items:
                    full_name = item.get("full_name")
                    if full_name:
                        repo_path = f"github.com/{full_name}"
                        if repo_path not in seen_repos:
                            seen_repos.add(repo_path)
                            popular_packages.append(repo_path)
                            stars = item.get("stargazers_count", 0)
                            self.logger.info(f"Found Go package: {repo_path} (Stars: {stars})")
                            
                            if len(popular_packages) >= self.max_packages:
                                break
                
                # Check if we've reached the end of results
                total_count = response.get("total_count", 0)
                if page * 100 >= total_count:
                    self.logger.info(f"Reached end of results ({total_count} total repositories)")
                    break
                
                page += 1
                # Be nice to GitHub API rate limits
                time.sleep(1.5)
            
            self.logger.info(f"Found total of {len(popular_packages)} unique Go packages")
            return popular_packages[:self.max_packages]
            
        except Exception as e:
            self.logger.error(f"Error fetching popular Go packages: {e}")
            self.logger.error(traceback.format_exc())
            return []
    
    def fetch_package_details(self, package_path: str) -> Dict[str, Any]:
        package_data = {
            "path": package_path,
            "name": package_path.split("/")[-1],
            "description": None,
            "stars": 0,
            "repository": f"https://{package_path}",
            "version": "",
            "license": ""
        }
        
        # If the package is from GitHub, get repository information directly
        if package_path.startswith("github.com/"):
            try:
                # Extract repository path (owner/repo)
                repo_path = "/".join(package_path.split("/")[1:3])  # Take first two parts after github.com
                if repo_path:
                    github_api_url = f"https://api.github.com/repos/{repo_path}"
                    
                    # Add GitHub token for authentication
                    token = next(self.token_iterator)
                    headers = {"Accept": "application/vnd.github.v3+json"}
                    if token:
                        headers["Authorization"] = f"token {token}"
                    
                    self.logger.info(f"Fetching GitHub data from: {github_api_url}")
                    github_response = self.make_request(github_api_url, headers=headers)
                    
                    if isinstance(github_response, dict):
                        package_data["stars"] = github_response.get("stargazers_count", 0)
                        package_data["description"] = github_response.get("description")
                        package_data["repository"] = github_response.get("html_url", package_data["repository"])
                        package_data["license"] = github_response.get("license", {}).get("name")
                        package_data["created_at"] = github_response.get("created_at")
                        package_data["updated_at"] = github_response.get("updated_at")
                        package_data["homepage"] = github_response.get("homepage")
                        package_data["language"] = github_response.get("language")
                        
                        # Get topics (keywords)
                        topics_url = f"{github_api_url}/topics"
                        topics_response = self.make_request(topics_url, headers=headers)
                        if isinstance(topics_response, dict) and "names" in topics_response:
                            package_data["keywords"] = topics_response.get("names", [])
                            
                        # Get owner information
                        owner = github_response.get("owner", {})
                        if owner and "login" in owner:
                            package_data["authors"] = [owner.get("login")]
            except Exception as e:
                self.logger.error(f"Error fetching GitHub data: {e}")
        
        # Try to fetch additional metadata from pkg.go.dev
        try:
            url = f"{self.pkg_url}/{package_path}"
            response = self.make_request(url)
            
            # If we got HTML content from pkg.go.dev, parse it for version info
            if "html_content" in response:
                html_content = response.get("html_content", "")
                soup = BeautifulSoup(html_content, "html.parser")
                
                # Extract version
                version_elem = soup.select_one(".go-Main-headerBreadcrumb span")
                if version_elem:
                    version_text = version_elem.get_text(strip=True)
                    if version_text.startswith("v"):
                        package_data["version"] = version_text
        except Exception as e:
            self.logger.error(f"Error fetching Go package details: {e}")
        
        return package_data
    
    def process_package(self, package_data: Dict) -> PackageData:
        return PackageData(
            id=package_data.get("path", ""),
            name=package_data.get("name", ""),
            language="go",
            description=package_data.get("description", ""),
            downloads=0,  # Not available
            stars=package_data.get("stars", 0),
            created_at=package_data.get("created_at"),
            updated_at=package_data.get("updated_at"),
            version=package_data.get("version", ""),
            repository=package_data.get("repository", ""),
            license=package_data.get("license", ""),
            homepage=package_data.get("homepage", ""),
            keywords=package_data.get("keywords", []),
            authors=package_data.get("authors", [])
        )
    
    def crawl_all(self):
        try:
            # Fetch list of popular packages
            popular_packages = self.fetch_popular_packages()
            self.logger.info(f"Found {len(popular_packages)} Go packages")
            
            # Process packages in batches using parallel processing
            batch_size = 10
            num_workers = min(10, len(popular_packages))
            
            for i in range(0, len(popular_packages), batch_size):
                batch = popular_packages[i:i+batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(popular_packages) + batch_size - 1)//batch_size}")
                
                packages = []
                
                # Process packages in parallel
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(self.fetch_package_details, package_path): package_path for package_path in batch}
                    
                    for future in as_completed(futures):
                        package_path = futures[future]
                        try:
                            package_data = future.result()
                            packages.append(self.process_package(package_data))
                            self.logger.info(f"Processed {package_path}")
                        except Exception as e:
                            self.logger.error(f"Error processing {package_path}: {e}")
                
                # Save batch of packages
                if packages:
                    self.save_package_batch(packages, "go")
                
                # Sleep a bit between batches to avoid overwhelming APIs
                time.sleep(2)
                
                if self.total_collected >= self.max_packages:
                    break
                
        except Exception as e:
            self.logger.error(f"Error during crawling Go packages: {e}")
            self.logger.error(traceback.format_exc())
        
        self.logger.info(f"Go packages crawling completed. Total collected: {self.total_collected}")
        return self.total_collected

class JavaMavenCrawler(BasePackageCrawler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Updated Maven Repository API with correct URL
        self.search_url = "https://search.maven.org/solrsearch/select"
    
    def search_packages(self, page: int = 0) -> Dict:
        params = {
            "q": "a:*",  # Search for artifacts with any name
            "rows": 20,
            "start": page * 20,
            "wt": "json"
        }
        response = self.make_request(self.search_url, params)
        if not response:
            self.logger.error("Failed to fetch Maven packages from search API")
        return response
    
    def process_package(self, package_data: Dict) -> PackageData:
        # Extract basic information from search response
        doc = package_data.get('doc', {})
        
        group_id = doc.get('g', '')
        artifact_id = doc.get('a', '')
        maven_id = f"{group_id}:{artifact_id}"
        
        # Get version and other metadata
        version = doc.get('latestVersion', '') or doc.get('v', '')
        timestamp = doc.get('timestamp')
        created_at = datetime.fromtimestamp(timestamp / 1000).isoformat() if timestamp else None
        
        return PackageData(
            id=maven_id,
            name=artifact_id,
            language="java",
            description=None,  # Not provided in basic search
            downloads=doc.get('download_count', 0) or 0,
            stars=0,  # Not available from Maven
            created_at=created_at,
            updated_at=created_at,  # Using same timestamp as created_at
            version=version,
            repository=None,  # Not directly provided
            license=None,  # Not in basic search response
            homepage=None,  # Not in basic search response
            keywords=[],
            authors=[]
        )
    
    def crawl_all(self):
        try:
            page = 0
            
            packages = []
            while self.total_collected < self.max_packages:
                search_results = self.search_packages(page=page)
                response = search_results.get('response', {})
                docs = response.get('docs', [])
                
                if not docs:
                    self.logger.warning("No more Maven packages found in search results")
                    # If we can't get packages from API, use a fallback list
                    if self.total_collected == 0:
                        self.logger.info("Using fallback Java packages")
                        fallback_packages = [
                            {"doc": {"g": "org.springframework", "a": "spring-core", "latestVersion": "5.3.20"}},
                            {"doc": {"g": "com.google.guava", "a": "guava", "latestVersion": "31.1-jre"}},
                            {"doc": {"g": "org.apache.commons", "a": "commons-lang3", "latestVersion": "3.12.0"}},
                            {"doc": {"g": "junit", "a": "junit", "latestVersion": "4.13.2"}},
                            {"doc": {"g": "org.slf4j", "a": "slf4j-api", "latestVersion": "1.7.36"}},
                            {"doc": {"g": "com.fasterxml.jackson.core", "a": "jackson-databind", "latestVersion": "2.13.3"}},
                            {"doc": {"g": "org.projectlombok", "a": "lombok", "latestVersion": "1.18.24"}},
                            {"doc": {"g": "org.mockito", "a": "mockito-core", "latestVersion": "4.6.1"}},
                            {"doc": {"g": "org.hibernate", "a": "hibernate-core", "latestVersion": "5.6.9.Final"}},
                            {"doc": {"g": "mysql", "a": "mysql-connector-java", "latestVersion": "8.0.29"}}
                        ]
                        for package_data in fallback_packages[:self.max_packages]:
                            packages.append(self.process_package(package_data))
                            
                        if packages:
                            self.save_package_batch(packages, "java")
                    break
                
                for doc in docs:
                    if self.total_collected >= self.max_packages:
                        break
                    
                    # Create package data directly from search result
                    package_data = {'doc': doc}
                    packages.append(self.process_package(package_data))
                    
                    if len(packages) >= 20:  # Save in smaller batches
                        self.save_package_batch(packages, "java")
                        packages = []
                
                page += 1
                time.sleep(1.0)  # Be more gentle with the server
            
            # Save any remaining packages
            if packages:
                self.save_package_batch(packages, "java")
                
        except Exception as e:
            self.logger.error(f"Error during crawling Java packages: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Use fallback if we have an error and no packages
            if self.total_collected == 0:
                self.logger.info("Using fallback Java packages after error")
                packages = []
                fallback_packages = [
                    {"doc": {"g": "org.springframework", "a": "spring-core", "latestVersion": "5.3.20"}},
                    {"doc": {"g": "com.google.guava", "a": "guava", "latestVersion": "31.1-jre"}},
                    {"doc": {"g": "org.apache.commons", "a": "commons-lang3", "latestVersion": "3.12.0"}},
                    {"doc": {"g": "junit", "a": "junit", "latestVersion": "4.13.2"}},
                    {"doc": {"g": "org.slf4j", "a": "slf4j-api", "latestVersion": "1.7.36"}},
                    {"doc": {"g": "com.fasterxml.jackson.core", "a": "jackson-databind", "latestVersion": "2.13.3"}},
                    {"doc": {"g": "org.projectlombok", "a": "lombok", "latestVersion": "1.18.24"}},
                    {"doc": {"g": "org.mockito", "a": "mockito-core", "latestVersion": "4.6.1"}},
                    {"doc": {"g": "org.hibernate", "a": "hibernate-core", "latestVersion": "5.6.9.Final"}},
                    {"doc": {"g": "mysql", "a": "mysql-connector-java", "latestVersion": "8.0.29"}}
                ]
                for package_data in fallback_packages[:self.max_packages]:
                    packages.append(self.process_package(package_data))
                    
                if packages:
                    self.save_package_batch(packages, "java")
        
        self.logger.info(f"Java packages crawling completed. Total collected: {self.total_collected}")
        return self.total_collected

class CPlusPlusConanCrawler(BasePackageCrawler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Since the Conan API is not easily accessible, use a static list of popular packages
        self.popular_packages = [
            {
                "name": "boost",
                "description": "Free peer-reviewed portable C++ source libraries",
                "versions": ["1.83.0"],
                "license": "BSL-1.0",
                "website": "https://www.boost.org",
                "downloads": 100000
            },
            {
                "name": "zlib",
                "description": "A massively spiffy yet delicately unobtrusive compression library",
                "versions": ["1.2.13"],
                "license": "Zlib",
                "website": "https://zlib.net",
                "downloads": 90000
            },
            {
                "name": "eigen",
                "description": "C++ template library for linear algebra",
                "versions": ["3.4.0"],
                "license": "MPL-2.0",
                "website": "https://eigen.tuxfamily.org",
                "downloads": 80000
            },
            {
                "name": "fmt",
                "description": "A modern formatting library",
                "versions": ["10.0.0"],
                "license": "MIT",
                "website": "https://fmt.dev",
                "downloads": 70000
            },
            {
                "name": "catch2",
                "description": "A modern, C++-native, test framework for unit-tests",
                "versions": ["3.3.2"],
                "license": "BSL-1.0",
                "website": "https://github.com/catchorg/Catch2",
                "downloads": 60000
            },
            {
                "name": "nlohmann_json",
                "description": "JSON for Modern C++",
                "versions": ["3.11.2"],
                "license": "MIT",
                "website": "https://github.com/nlohmann/json",
                "downloads": 50000
            },
            {
                "name": "opencv",
                "description": "Open Source Computer Vision Library",
                "versions": ["4.7.0"],
                "license": "Apache-2.0",
                "website": "https://opencv.org",
                "downloads": 40000
            },
            {
                "name": "openssl",
                "description": "Secure Socket Layer library",
                "versions": ["3.1.0"],
                "license": "OpenSSL",
                "website": "https://www.openssl.org",
                "downloads": 30000
            },
            {
                "name": "protobuf",
                "description": "Protocol Buffers - Google's data interchange format",
                "versions": ["3.21.12"],
                "license": "BSD-3-Clause",
                "website": "https://developers.google.com/protocol-buffers",
                "downloads": 25000
            },
            {
                "name": "poco",
                "description": "Modern, powerful open source C++ class libraries",
                "versions": ["1.12.4"],
                "license": "BSL-1.0",
                "website": "https://pocoproject.org",
                "downloads": 20000
            },
            {
                "name": "spdlog",
                "description": "Fast C++ logging library",
                "versions": ["1.11.0"],
                "license": "MIT",
                "website": "https://github.com/gabime/spdlog",
                "downloads": 15000
            },
            {
                "name": "sqlite3",
                "description": "Self-contained, serverless, zero-configuration SQL database engine",
                "versions": ["3.41.2"],
                "license": "Public Domain",
                "website": "https://www.sqlite.org",
                "downloads": 10000
            }
        ]
    
    def process_package(self, package_data: Dict) -> PackageData:
        # Extract information from the static package data
        name = package_data.get("name", "")
        description = package_data.get("description", "")
        
        # Get version and license
        versions = package_data.get("versions", [])
        latest_version = versions[0] if versions else ""
        
        license_str = package_data.get("license", "")
        if isinstance(license_str, list):
            license_str = ", ".join(license_str)
        
        return PackageData(
            id=name,
            name=name,
            language="c++",
            description=description,
            downloads=package_data.get("downloads", 0),
            stars=0,  # Not provided
            created_at=None,  # Not provided
            updated_at=None,  # Not provided
            version=latest_version,
            repository=package_data.get("repository") or package_data.get("website"),
            license=license_str,
            homepage=package_data.get("website"),
            keywords=[],  # Not typically provided
            authors=[]  # Not easily extracted
        )
    
    def crawl_all(self):
        try:
            packages = []
            self.logger.info("Using static list of popular C++ packages")
            
            # Use the static list of popular packages
            for package_data in self.popular_packages[:self.max_packages]:
                if self.total_collected >= self.max_packages:
                    break
                
                self.logger.info(f"Processing C++ package: {package_data['name']}")
                packages.append(self.process_package(package_data))
                
                if len(packages) >= 5:  # Save in smaller batches
                    self.save_package_batch(packages, "c++")
                    packages = []
                
                time.sleep(0.1)  # Short delay between processing
            
            # Save any remaining packages
            if packages:
                self.save_package_batch(packages, "c++")
                
        except Exception as e:
            self.logger.error(f"Error during crawling C++ packages: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        self.logger.info(f"C++ packages crawling completed. Total collected: {self.total_collected}")
        return self.total_collected
    
class MetaPackageCrawler:
    def __init__(
        self,
        output_dir: str = "package_data",
        languages: List[str] = None,
        max_packages_per_lang: int = 1000,
        max_workers: int = 3
    ):
        self.output_dir = output_dir
        self.languages = languages or ["python", "rust", "javascript", "go", "java", "c++"]
        self.max_packages_per_lang = max_packages_per_lang
        self.max_workers = max_workers
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def get_crawler_for_language(self, language: str) -> BasePackageCrawler:
        if language == "rust":
            return RustCratesCrawler(output_dir=self.output_dir, max_packages=self.max_packages_per_lang)
        elif language == "python":
            return PythonPyPICrawler(output_dir=self.output_dir, max_packages=self.max_packages_per_lang)
        elif language == "javascript":
            return JavaScriptNPMCrawler(output_dir=self.output_dir, max_packages=self.max_packages_per_lang)
        elif language == "go":
            return GoPackageCrawler(output_dir=self.output_dir, max_packages=self.max_packages_per_lang)
        elif language == "java":
            return JavaMavenCrawler(output_dir=self.output_dir, max_packages=self.max_packages_per_lang)
        elif language == "c++":
            return CPlusPlusConanCrawler(output_dir=self.output_dir, max_packages=self.max_packages_per_lang)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def crawl_all_languages(self):
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_lang = {
                executor.submit(self.crawl_language, lang): lang 
                for lang in self.languages
            }
            
            for future in as_completed(future_to_lang):
                language = future_to_lang[future]
                try:
                    count = future.result()
                    results[language] = count
                except Exception as e:
                    self.logger.error(f"Error crawling {language}: {e}")
                    results[language] = 0
        
        self.logger.info("All crawling completed")
        return results
    
    def crawl_language(self, language: str) -> int:
        try:
            self.logger.info(f"Starting crawler for {language}")
            crawler = self.get_crawler_for_language(language)
            count = crawler.crawl_all()
            self.logger.info(f"Completed crawler for {language}. Collected {count} packages")
            return count
        except Exception as e:
            self.logger.error(f"Error in {language} crawler: {e}")
            return 0

def main():
    crawler = MetaPackageCrawler(
        output_dir="package_data",
        max_packages_per_lang=10,
        max_workers=3
    )
    
    try:
        results = crawler.crawl_all_languages()
        print("Crawling completed with the following results:")
        for language, count in results.items():
            print(f"{language}: {count} packages")
    except KeyboardInterrupt:
        print("Crawling interrupted by user")
    except Exception as e:
        print(f"Error during crawling: {e}")

if __name__ == "__main__":
    main()