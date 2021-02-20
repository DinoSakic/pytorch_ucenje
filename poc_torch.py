import torch

x = torch.rand(1, requires_grad=True)


def mse(z):  # mean square error; trazimo 3
    # c = y - error; return (c^2)/2
    diff = 3.0 - z  # koliko smo udaljeni od 3; loss funkcija
    return (diff * diff).sum() / 2

learning_rate = 1e-3


for i in range(0, 10000):
    y = x + 1.0
    loss = mse(y)
    # ovdje se desava backpropagation gradijenta; loss je type tensor
    loss.backward()
    # ovdje se desava ucenje
    with torch.no_grad():
        # gradijent govori u kojem smjeru je od greske, zato bp ide u
        # suprotnom smijeru da ispravi gresku
        x -= learning_rate * x.grad
        x.grad.zero_()  # potrebno svaki put gradijente postaviti na 0 jer se akumuliraju
    if i % 10 == 0:
        print(f'{i/100}% | {x}')
