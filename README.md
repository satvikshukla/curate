# Curate

Python program to find the art movement and the artist the input image most probably would have come from. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

*This code has been tested under Python 3.6*

### Prerequisites

Here are the few packages needed to run the file `tests/predict.py`

```
pandas
numpy
pillow
tensorflow
h5py
scipy
pyyaml
keras
wikipedia
```

### Installing

To install the above libraries, go to terminal if on UNIX or bash if on Windows and type in the following command

```
pip install pandas numpy pillow h5py scipy pyyaml tensorflow keras wikipedia
```

Two more direcctories need to be created to carry out the tests. Inside the main repository directory, creates two directories named `images` and `data`. Inside `images` put all the image files that are needed to carry out the desired predictions. Inside `data` two models are needed to be stored to make the predictions. They can be downloaded from [here](). Inside `data`, one more file named `all_data_info.csv` is needed that can be downloaded from kaggle through [this link](https://www.kaggle.com/c/painter-by-numbers/data).

## Author

* **Satvik Shukla** - [satvikshukla](https://github.com/satvikshukla)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [Painter by numbers](https://www.kaggle.com/c/painter-by-numbers) competition for providing the data to make the models.
* [This webpage](https://harishnarayanan.org/writing/artistic-style-transfer/) for providing help with neural style transfer
* [This repository](https://github.com/kevinzakka/style-transfer) for acting as reference for style transfer