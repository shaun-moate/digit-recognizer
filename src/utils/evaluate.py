import torch
import torch.nn.functional as F
import json

def evaluate(network, loader, params, results=[], store=False):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(loader.dataset)
    results.append((test_loss, correct / len(loader.dataset)))

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(loader.dataset),
      100. * correct / len(loader.dataset)))
    if json:
        output = {"Avg. Loss": round(test_loss, 4), "Accuracy": round((correct / len(loader.dataset)).item(), 4)}
        output_file = open(params.eval.metrics_path, "w")
        json.dump(output, output_file, indent=6)
