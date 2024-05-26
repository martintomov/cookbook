## Methodology

- **Data Preprocessing**: The data is cleaned from punctuation and stopwords. The reviews are then indexed and converted to sequences of integers. The reviews are then padded to the same length.
- **RNN Model**: The model consists of an embedding layer, an RNN layer, and a linear layer. The output is passed through a sigmoid function.
- **LSTM Model**: The model consists of an embedding layer, an LSTM layer, and a linear layer. The output is passed through a sigmoid function.

## Testing

Each model is tested with different hyperparameters, including embedding dimensions, hidden dimensions, number of layers, dropout, learning rate, criterion, and optimizer. Detailed graphs for each test can be found in the notebook.

## Summary

For each model, among the tested hyperparameters, the best results were:

| Model | Test Accuracy       |
| ----- | ------------------- |
| RNN   | 0.478287841191067   |
| LSTM  | 0.49545078577336643 |

Which surpasses the required accuracy of 0.45.
