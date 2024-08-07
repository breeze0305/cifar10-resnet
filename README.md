# 基於ResNet-50的CIFAR-10圖像分類

此倉庫包含使用ResNet-50架構進行CIFAR-10圖像分類的實現。項目使用PyTorch框架，包含數據準備、模型訓練和評估的腳本。

## 目錄

- [安裝](#安裝)
- [數據集準備](#數據集準備)
- [使用方法](#使用方法)
  - [數據準備](#數據準備)
  - [模型訓練](#模型訓練)
  - [評估](#評估)
- [檔案描述](#檔案描述)
- [授權](#授權)

## 安裝

首先，將倉庫克隆到您的本地機器：

```bash
git clone https://github.com/yourusername/cifar10-classification.git
cd cifar10-classification
```

確保您已安裝Python 3.8或更高版本。您可以使用`conda`來設置虛擬環境：

```bash
conda create -n cifar python=3.8
conda activate cifar  
```

安裝所需的依賴包：

```bash
pip install -r requirements.txt
```

## 數據集準備

此項目使用CIFAR-10數據集，該數據集包含60,000張32x32的彩色圖像，分為10個類別，每個類別有6,000張圖像。
[cifar-10官方連結](https://www.cs.toronto.edu/~kriz/cifar.html)


### 步驟1：下載數據集

從[這裡](https://www.cs.toronto.edu/~kriz/cifar.html)下載CIFAR-10數據集。

### 步驟2：解壓數據集

使用提供的腳本解壓數據集：

```bash
python unpake_cifar10.py
```

這將把數據集組織到`train`和`val`目錄中，每個類別都有一個子文件夾。

## 使用方法

### 數據準備

在訓練之前，請確保數據已正確解壓並通過以下指令進行整理：
```bash
python unpake_cifar10.py
```

### 模型訓練

您可以使用以下命令在CIFAR-10數據集上訓練ResNet-50模型：

```bash
python main_train.py
```

該腳本將自動執行以下操作：
- 加載CIFAR-10數據集。
- 應用數據增強。
- 訓練ResNet-50模型。
- 使用Weights & Biases記錄訓練過程。

### 評估

模型在每個epoch後都會在測試集上進行評估。結果（包括損失和準確率）會被記錄下來，並且最佳模型權重會被保存。

## 檔案描述

- `test_cuda.py`: 檢查是否有CUDA兼容的GPU可用，並返回合適的設備（GPU或CPU）。
- `unpake_cifar10.py`: 負責解壓和組織CIFAR-10數據集到訓練和驗證目錄中。
- `main_dataset.py`: 包含自定義的PyTorch數據集`Cifar10Dataset`類，用於加載和處理CIFAR-10圖像，並提供可選的數據轉換。
- `main_train.py`: 用於在CIFAR-10數據集上訓練ResNet-50模型的主腳本，包括數據加載、訓練循環和驗證。

## 授權

本項目基於MIT許可證進行授權 - 詳見[LICENSE](LICENSE)文件。
