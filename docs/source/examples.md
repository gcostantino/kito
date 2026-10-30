# Examples

The `examples/` directory of the repository contains two complete scripts. Both download MNIST to `./data`,
train with `Engine.fit`, load the best checkpoint and run `Engine.predict` on 1000 test images.

| Script | What it shows |
|---|---|
| [`mnist_simple.py`](https://github.com/gcostantino/kito/blob/main/examples/mnist_simple.py) | Feed-forward classifier, the smallest complete workflow |
| [`mnist_cnn.py`](https://github.com/gcostantino/kito/blob/main/examples/mnist_cnn.py) | CNN with data augmentation, weight decay and `EarlyStoppingCallback` |

Run them from the `examples/` directory:

```bash
cd examples
python mnist_simple.py
```

Checkpoints and logs are written to `examples/outputs`.
