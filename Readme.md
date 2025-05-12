<h1>🌐 Domain-Adversarial Neural Network (DANN) for Domain Adaptation - 1D Signal</h1>

<p>
This project implements a <strong>DANN (Domain-Adversarial Neural Network)</strong> 
to perform robust fault classification under domain shift using the bearing dataset.
</p>

<hr/>

<h2>📁 Project Structure</h2>

<pre>
dann_pipeline/
├── main.py              # Entry point (load, train, evaluate)
├── config.py            # Argument parser
├── dataset.py           # Source/target dataloaders
├── model.py             # DANN architecture (feature extractor + classifiers)
├── train.py             # Adversarial training routine
├── test.py              # Final evaluation on target domain
├── utils.py             # Optional tools (e.g. plotting)
├── data/                # .npy input files
└── result/              # model.pt, logs, loss curves
</pre>

<hr/>
<h2>📥 Dataset</h2>

<p>
This project uses two publicly available bearing fault datasets adapted for domain adaptation experiments:
</p>

<ul>
  <li>
    <strong>CWRU Bearing Dataset</strong> – Provided by Case Western Reserve University<br/>
    🔗 <a href="https://engineering.case.edu/bearingdatacenter/download-data-file" target="_blank">
    https://engineering.case.edu/bearingdatacenter/download-data-file
    </a>
  </li>
  <br/>
  <li>
    <strong>IMS Bearing Dataset</strong> – Provided via Data.gov<br/>
    🔗 <a href="https://catalog.data.gov/dataset/ims-bearings" target="_blank">
    https://catalog.data.gov/dataset/ims-bearings
    </a>
  </li>
</ul>

<p>
For signal preprocessing and conversion into <code>.npy</code> format, refer to the preprocessing code in this repository:<br/>
🔧 <a href="https://github.com/97yong/signal-fault-classification" target="_blank">
https://github.com/97yong/signal-fault-classification
</a>
</p>

<p>
Preprocessed data is saved in the following format as <code>.npy</code>:
</p>

<pre>
data/
├── X_source.npy
├── Y_source.npy
├── X_target.npy
├── Y_target.npy
├── X_target_test.npy
└── Y_target_test.npy
</pre>

<hr/>

<h2>⚙️ Training Options</h2>

<p>You can configure training parameters via <code>config.py</code> or pass them through <code>opt</code>.</p>

<table>
  <tr><th>Argument</th><th>Description</th><th>Default</th></tr>
  <tr><td><code>--epochs</code></td><td>Number of training epochs</td><td>10</td></tr>
  <tr><td><code>--lr</code></td><td>Learning rate</td><td>1e-4</td></tr>
  <tr><td><code>--batch_size</code></td><td>Mini-batch size</td><td>64</td></tr>
</table>

<hr/>

<h2>🧠 Model Overview</h2>

<p>
DANN consists of:
</p>
<ul>
  <li><strong>Feature Extractor</strong> – 1D CNN layers</li>
  <li><strong>Label Classifier</strong> – Predicts class labels for source domain</li>
  <li><strong>Domain Classifier</strong> – Predicts domain (source/target) using GRL (gradient reversal)</li>
</ul>

<p>Training is done with <strong>adversarial loss</strong> to align the feature distributions.</p>

<hr/>

<h2>🚀 How to Run</h2>

<pre><code>pip install numpy torch scikit-learn tqdm matplotlib</code></pre>

<pre><code>python main.py</code></pre>

<p>This will:</p>
<ol>
  <li>Load source/target domain data</li>
  <li>Train a domain-adversarial model</li>
  <li>Evaluate performance on target test data</li>
</ol>

<hr/>

<h2> Views </h2>

![](http://profile-counter.glitch.me/97yong-domain-adaptation-signal/count.svg)

<h2>📚 Reference</h2>

<p>
Ganin, Y., & Lempitsky, V. (2015). <br/>
<strong>"Unsupervised Domain Adaptation by Backpropagation."</strong> <br/>
In Proceedings of the 32nd International Conference on Machine Learning (ICML), 1180–1189. <br/>
🔗 <a href="https://arxiv.org/abs/1409.7495" target="_blank">arXiv:1409.7495</a>
</p>

<hr/>
