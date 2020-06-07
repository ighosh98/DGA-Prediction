# DGA-Prediction
Detecting Domains Names generated by Non-classical Domain Generation Algorithms in Botnets
#### Paper Link: A Comparative Study of Various Neural Network-Based Learning Models for DGA Detection
#### DGAs and datasets used: 

##### Datasets
- Alexa top 1m domains
- The [Open-Source Intelligence (OSINT) DGA feed from Bambenek Consulting](http://osint.bambenekconsulting.com/feeds/), which provided the malicious domain names [31]. This data feed was based on 50 DGA algorithms that together contained 852,116 malicious domain names. The dataset was downloaded on May 23, 2018 and DGAs were generated on that day. Also, on April 18, 2019, an additional dataset of 855,197 DGA generated domains was downloaded from OSINT for testing differences in model performance based on time and is regarded as a separate test dataset.
DGAs to be implemented
- classical DGA domains for the following malware families: banjori, corebot, cryptolocker, dircrypt, kraken, lockyv2, pykspa, qakbot, ramdo, ramnit, and simda.
- Word-based/dictionary DGA domains for the following classical malware families: **[Done]**
	- gozi
	- matsnu
	- suppobox

- Word-based/dictionary DGA domains for the following classical malware families: **[To be Implemented]**
	- pizd DGA generator
	- nymaim2 DGA generator
	- cryptowall
	
**The other branches carry further works for word DGAs nymaim2 and pizd. They are named accordingly.**

#### dga_predict  default settings:
- training splits: 76% training, 4% validation, %20 testing
- all models were trained with a batch size of 128
- The CNN, LSTM, and CNN+LSTM models used up to 25 epochs, while the bigram models used up to 50 epochs.
## Environment Setup Script

```
conda create -n <ENVIRONMENT_NAME> python=2.7 scikit-learn keras tensorflow-gpu matplotlib
source activate <ENVIRONMENT_NAME>
pip install tldextract
```
### LICENSE
[Creative Commons Zero v1.0 Universal](https://github.com/ighosh98/DGA-Prediction/blob/master/LICENSE)
