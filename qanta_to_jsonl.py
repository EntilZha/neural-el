import json
import click
import tqdm

@click.command()
@click.argument('qanta_file')
@click.argument('out_file')
def main(qanta_file: str, out_file: str):
    with open(qanta_file) as f:
        questions = [q for q in json.load(f)['questions'] if q['page'] is not None]
    
    with open(out_file, 'w') as f:
        for q in tqdm.tqdm(questions):
            for sent_id, (start, end) in enumerate(q['tokenizations']):
                sentence = q['text'][start:end]
                f.write(json.dumps({
                    'text': sentence,
                    'qanta_id': q['qanta_id'],
                    'sent_id': sent_id
                }))
                f.write('\n')


if __name__ == '__main__':
    main()