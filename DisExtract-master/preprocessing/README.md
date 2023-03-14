
## Usage (BookCorpus)

### 1. Set up CoreNLP Server

DisExtract uses Stanford's CoreNLP dependency parser. Follow the instructions [https://github.com/erindb/corenlp-ec2-startup](https://github.com/erindb/corenlp-ec2-startup) to run the server. Keep this server running in the background. DisExtact scripts will make calls to this server.

### 2. Update `example_config.json` to point to appropriate data directories.

### 3. Filter sentence pairs containing discourse marker strings.

	python bookcorpus.py --filter

### 4. Parse the sentences to get sentence pairs separated by discourse markers. This script calls the CoreNLP Server.

	python bookcorpus.py --parse

At this point, you will have the file `corpus/bookcorpus/markers_ALL18/parsed_sentences_pairs/ALL18_parsed_sentence_pairs.txt` which contains tab-separated sentence pairs and the corresponding discourse marker linking them.

### 5. Finish preprocessing for DisSent:

python producer.py --data_file corpus/bookcorpus/markers_ALL18/parsed_sentences_pairs/ALL18_parsed_sentence_pairs.txt --out_prefix ALL18_2019jan02
