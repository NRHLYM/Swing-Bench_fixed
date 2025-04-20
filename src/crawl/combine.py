import requests
import re
import jsonlines
from tqdm import tqdm
from datasets import load_dataset
import concurrent.futures
import os
from bs4 import BeautifulSoup

def extract_ci_name_list(pull: dict) -> list[str]:
    # TODO(wdxu): adapt CIs which are not configured in .github/workflow/
    print('processing {} {}'.format(pull['base']['repo']['full_name'], pull['number']))
    checks_info_ptn = 'https://github.com/{}/pull/{}/checks'
    checks_url = checks_info_ptn.format(pull['base']['repo']['full_name'], pull['number'])
    print(checks_url)
    response = requests.get(checks_url)
    if response.status_code == 200:
        runs_url_prefix = '{}/actions/runs/'.format(pull['base']['repo']['full_name'])
        runs_url_ptn = re.compile(rf'{runs_url_prefix}(\d+)')
        matches = runs_url_ptn.findall(response.text)
        # print(matches)
        ci_names = []
        for num in list(set(matches)):
            run_url = 'https://github.com/{}/actions/runs/{}/workflow'.format(pull['base']['repo']['full_name'], num)
            run_response = requests.get(run_url)
            if run_response.status_code == 200:
                run_soup = BeautifulSoup(run_response.text, "html.parser")
                yml = run_soup.find_all('table')
                assert len(yml) == 1
                yml = yml[0].get('data-tagsearch-path')
                action_list = run_soup.find_all("a", class_="ActionListContent--visual16")
                summary_index = None
                usage_index = None
                for i, item in enumerate(action_list):
                    href = item.get('href', '')
                    if '/actions/runs/' in href and href.endswith(str(num)):
                        summary_index = i
                    elif href.endswith('/usage'):
                        usage_index = i
                
                if summary_index is None or usage_index is None or summary_index >= usage_index:
                    raise ValueError("Could not find Summary or Usage in the expected order")
                
                action_list = action_list[summary_index + 1:usage_index]
                ci = None
                for action in action_list:
                    svg = action.find('svg', attrs={'aria-label': lambda value: value != 'skipped: '})
                    if svg:
                        label_span = action.find('span', class_='ActionListItem-label')
                        if label_span:
                            ci = label_span.get_text(strip=True)
                assert ci is not None
                ci_names.append((ci, yml))

        return list(set(ci_names))
    return []

def process_item(d):
    repo = d["repo"].split("/")[1]
    pull_number = d["instance_id"].split("-")[-1]
    try:
        if os.path.exists(f"/raid/Swing-Bench/src/crawl/issues/prs/{repo}-prs.jsonl"):
            with jsonlines.open(f"/raid/Swing-Bench/src/crawl/issues/prs/{repo}-prs.jsonl", "r") as f:
                prs = list(f)
        else:
            with jsonlines.open(f"/raid/Swing-Bench/src/crawl/issues/prs/{repo}-prs-20180101.jsonl", "r") as f:
                prs = list(f)
        prs = [pr for pr in prs if str(pr["number"]) == pull_number]
        assert len(prs) == 1, f'len(prs) = {len(prs)}, {repo}'
        pr = prs[0]
        d["merge_commit_sha"] = pr["merge_commit_sha"]
        try:
            d["created_at"] = d["created_at"].strftime("%Y-%m-%d")
        except:
            pass
        d["ci_name_list"] = extract_ci_name_list(pr)
        return d
    except Exception as e:
        print(f"Error processing {repo}-{pull_number}: {str(e)}")
        return None

def main():
    # used = load_dataset("SwingBench/SWE-Rust")["train"]
    # used = [d['instance_id'] for d in used]
    used = []
    # ds = load_dataset("SwingBench/SWE-Rust")["train"]
    with jsonlines.open("issues/all_tasks.jsonl", "r") as f:
        ds = list(f)
    print(len(ds))
    ds = [d for d in ds if not d['instance_id'] in used]
    print(len(ds))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_item, d) for d in ds]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                with jsonlines.open("tasks_with_ci_rest_0318.jsonl", "a") as f:
                    f.write(result)