# Antimicrobial Peptide Sequence Generation

This project aims to develop a deep learning model using PyTorch for generating more potent antimicrobial peptides. Antimicrobial peptides (AMPs) are short sequences of amino acids that exhibit strong antimicrobial properties against a wide range of pathogens, making them promising candidates for novel antimicrobial therapies.

The main objective of this project is to train a Long Short-Term Memory (LSTM) model, a type of Recurrent Neural Network (RNN), to learn from a dataset of known antimicrobial peptides and generate novel peptide sequences with enhanced antimicrobial activity. By leveraging the power of deep learning and sequence generation techniques, we seek to explore the vast space of peptide sequences and discover new potential candidates with improved efficacy against pathogens.

The model will be trained using a training dataset of antimicrobial peptides, and the generated sequences will be evaluated based on their predicted antimicrobial properties. The project includes hyperparameter tuning to optimize the model's performance and generate high-quality sequences.


## Project Structure

The project structure is organized as follows:
- data/
    - [sequences.csv](data/sequences.csv)
- src/
    - [utils.py](src/utils.py)
    - [dataset.py](src/dataset.py)
    - [train.py](src/train.py)
    - [argument_parser.py](src/argument_parser.py)
- models/
    - [models.py](models/models.py)
- notebooks/
    - [param_search.ipynb](notebooks/param_search.py)
- reports/
    - ...
- [main.py](main.py)
- [setup.py](setup.py)
- [requirements.txt](requirements.txt)
- [LICENSE](LICENSE)
- [README.md](README)

## Usage

Please follow the instructions below to set up and execute the project:

1. Clone the directory to your computer by running `git clone https://github.com/sphamtambo/peptides_gen`
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Customize the model parameters in the [argument_parser.py](src/argument_parser.py) file to meet your specific requirements.
4. Run the project by executing python [main.py](main.py) in the terminal.

	- Pretraining:

	```bash
	python main.py --mode pretrain --session_name train --num_epochs 100 --dataset sequences.csv  
	```

	- Finetuning on a different dataset:

	```bash
	python main.py --mode finetune --session_name finetune --dataset new_data.csv --checkpoint reports/pretrain/model_best.h5
	```

	- Sampling from a pretrained model:

	```bash
	python main.py --mode sample --session_name sample --checkpoint reports/train/model_best.h5
	```

	- Cross-validation and hyperparameter search:

	```
	Adjust the hyperparameters accordingly in the param_search.ipynb notebook.
	```

5. Monitor the training progress and access the generated reports in the reports/ directory.

## License

This project is licensed under the [MIT License](LICENSE).
