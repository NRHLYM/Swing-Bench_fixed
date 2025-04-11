import os
import json
import time
import random
import logging
import argparse
import asyncio
import aiohttp
import aiofiles
import subprocess
from tqdm import tqdm
from typing import List, Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("github_crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com"
LANGUAGES = [
    "JavaScript", "Python", "Java", 
    "Go", "C++", "TypeScript", "Rust", "PHP",
    "C#", "Ruby", "Swift", "Kotlin"
]

class GitHubCrawler:
    def __init__(
        self, 
        output_dir: str = "github_repos", 
        max_repos_per_lang: int = 100,
        concurrency: int = 10,
        retry_limit: int = 3,
        timeout: int = 30
    ):
        self.tokens = os.environ.get("GITHUB_TOKENS").split(",")
        self.output_dir = output_dir
        self.max_repos_per_lang = max_repos_per_lang
        self.concurrency = concurrency
        self.retry_limit = retry_limit
        self.timeout = timeout
        self.progress_file = os.path.join(output_dir, "progress.json")
        self.semaphore = asyncio.Semaphore(concurrency)
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.progress = self._load_progress()
    
    def _load_progress(self) -> Dict[str, Any]:
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load progress file: {e}")
        
        return {
            "completed_languages": [],
            "current_language": None,
            "current_page": 1,
            "repos_collected": 0,
            "last_updated": time.time()
        }
    
    async def _save_progress(self) -> None:
        self.progress["last_updated"] = time.time()
        async with aiofiles.open(self.progress_file, 'w') as f:
            await f.write(json.dumps(self.progress, indent=2))
    
    async def _get_random_token(self) -> str:
        return random.choice(self.tokens)
    
    async def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        async with self.semaphore:
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"token {await self._get_random_token()}"
            }
            
            for attempt in range(self.retry_limit):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            url, 
                            headers=headers, 
                            params=params,
                            timeout=self.timeout
                        ) as response:
                            if response.status == 200:
                                return await response.json()
                            elif response.status == 403:
                                reset_time = int(response.headers.get('X-RateLimit-Reset', 0))
                                wait_time = max(1, reset_time - time.time())
                                logger.warning(f"API rate limit reached, waiting {wait_time:.2f} seconds before retrying")
                                await asyncio.sleep(wait_time + random.uniform(0.1, 2.0))
                            elif response.status == 404:
                                logger.error(f"Resource not found: {url}")
                                return {}
                            else:
                                logger.error(f"Request failed: {response.status}, {await response.text()}")
                                await asyncio.sleep(1 * (attempt + 1))
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.error(f"Request exception: {e}, attempt: {attempt+1}/{self.retry_limit}")
                    await asyncio.sleep(1 * (attempt + 1))
            
            logger.error(f"Request failed, max retries reached: {url}")
            return {}
    
    async def _fetch_repositories(self, language: str, page: int = 1) -> List[Dict]:
        url = f"{GITHUB_API_URL}/search/repositories"
        params = {
            "q": f"language:{language}",
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": page
        }
        
        data = await self._make_request(url, params)
        return data.get("items", [])
    
    async def _save_repository_data(self, language: str, repos: List[Dict]) -> None:
        jsonl_file_path = os.path.join(self.output_dir, "repositories.jsonl")
        
        async with aiofiles.open(jsonl_file_path, 'a') as jsonl_file:
            for repo in repos:
                repo_data = {
                    "name": repo.get("name"),
                    "url": repo.get("html_url"),
                    "language": language,
                    "stars": repo.get("stargazers_count")
                }
                
                await jsonl_file.write(json.dumps(repo_data) + "\n")
                
                self._clone_repository(repo_data["url"])

    def _clone_repository(self, repo_url: str) -> None:
        subprocess.Popen(["git", "clone", repo_url], cwd=self.output_dir)
    
    async def crawl_language(self, language: str) -> None:
        if language in self.progress["completed_languages"]:
            logger.info(f"Language {language} already crawled, skipping")
            return

        start_page = 1
        repos_collected = 0

        if language == self.progress["current_language"]:
            start_page = self.progress["current_page"]
            repos_collected = self.progress["repos_collected"]
            logger.info(f"Resuming crawl for {language}, page: {start_page}, already collected: {repos_collected}")
        else:
            self.progress["current_language"] = language
            self.progress["current_page"] = start_page
            self.progress["repos_collected"] = repos_collected
            await self._save_progress()
        
        page = start_page
        pbar = tqdm(total=self.max_repos_per_lang, initial=repos_collected, desc=f"Crawling {language}")
        
        while repos_collected < self.max_repos_per_lang:
            repos = await self._fetch_repositories(language, page)
            
            if not repos:
                logger.info(f"No more repositories for language {language}")
                break
            
            remaining = self.max_repos_per_lang - repos_collected
            repos_to_save = repos[:remaining]
            
            await self._save_repository_data(language, repos_to_save)
            
            repos_collected += len(repos_to_save)
            pbar.update(len(repos_to_save))
            
            page += 1
            self.progress["current_page"] = page
            self.progress["repos_collected"] = repos_collected
            await self._save_progress()
            
            if repos_collected >= self.max_repos_per_lang or len(repos) < 100:
                break
            
            await asyncio.sleep(random.uniform(0.5, 2.0))
        
        pbar.close()
        
        self.progress["completed_languages"].append(language)
        self.progress["current_language"] = None
        self.progress["current_page"] = 1
        self.progress["repos_collected"] = 0
        await self._save_progress()
        
        logger.info(f"Finished crawling {language}, collected {repos_collected} repositories")
    
    async def crawl_all_languages(self) -> None:
        for language in LANGUAGES:
            await self.crawl_language(language)
            await asyncio.sleep(random.uniform(1.0, 3.0))
    
    async def crawl_parallel(self, languages: List[str] = None) -> None:
        if not languages:
            languages = [lang for lang in LANGUAGES if lang not in self.progress["completed_languages"]]
        
        logger.info(f"Preparing to crawl {len(languages)} languages in parallel")
        tasks = [self.crawl_language(language) for language in languages]
        await asyncio.gather(*tasks)
        
        logger.info("All languages crawled successfully!")

async def main():
    parser = argparse.ArgumentParser(description="GitHub Repository Crawler")
    parser.add_argument("--output", type=str, default="github_repos", help="Output directory")
    parser.add_argument("--max-repos", type=int, default=100, help="Maximum number of repositories to crawl per language")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--languages", type=str, help="Languages to crawl (comma-separated), all supported languages if not specified")
    parser.add_argument("--parallel", action="store_true", help="Whether to crawl multiple languages in parallel")
    
    args = parser.parse_args()
    
    languages_to_crawl = None
    if args.languages:
        languages_to_crawl = [lang.strip() for lang in args.languages.split(",")]
    
    crawler = GitHubCrawler(
        output_dir=args.output,
        max_repos_per_lang=args.max_repos,
        concurrency=args.concurrency
    )
    
    if args.parallel:
        await crawler.crawl_parallel(languages_to_crawl)
    else:
        if languages_to_crawl:
            for language in languages_to_crawl:
                await crawler.crawl_language(language)
        else:
            await crawler.crawl_all_languages()

if __name__ == "__main__":
    asyncio.run(main())

# python -m src.crawl.crawl_all \
#   --output "github_repos" \
#   --max-repos 10 \
#   --concurrency 10 \
#   --languages "Python,JavaScript" \
#   --parallel