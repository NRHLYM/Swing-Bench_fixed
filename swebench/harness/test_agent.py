# import swebench.harness.agent as agent


if __name__ == "__main__":
    gen_output_path = '/mnt/Data/wdxu/github/Swing-Bench/tmpdata/agent_gen_output'
    patch_response_list = []
    test_response_list = []
    with open(gen_output_path, 'r') as f:
        raw_lines = f.readlines()
        raw_lines = [line.strip() for line in raw_lines]
        concatenated_text = '\n'.join(raw_lines)
        concatenated_text = concatenated_text.split('patch response')
        concatenated_text = [item for item in concatenated_text if item.strip()]
        for txt in concatenated_text:
            patch_response, test_response = txt.split('test response')
            if '```patch' not in test_response:
                print(test_response)
            patch_response_list.append(patch_response)
            test_response_list.append(test_response)

    # print(patch_response_list[0])
    # print(test_response_list[0])
