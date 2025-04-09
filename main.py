from arguments import get_args
from dataset import get_dataloaders
from train import train_model
from test import evaluate

def main():
    opt = get_args()

    data_path = './data'
    source_X_path = data_path + '/X_source.npy'
    source_Y_path = data_path + '/Y_source.npy'
    target_X_path = data_path + '/X_target.npy'
    target_Y_path = data_path + '/Y_target.npy'
    target_X_test_path = data_path + '/X_target_test.npy'
    target_Y_test_path = data_path + '/Y_target_test.npy'

    source_dl, target_dl, target_test_dl = get_dataloaders(opt, source_X_path, target_X_path, target_X_test_path, source_Y_path, target_Y_path, target_Y_test_path)

    extractor, predictor = train_model(opt, source_dl, target_dl)
    acc = evaluate(target_test_dl, extractor, predictor)
    print(f"✅ Test Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    main()