# DGA-Prediction
Detecting Domains Names generated by Non-classical Domain Generation Algorithms in Botnets
## DGA Details
#### DGAs and datsets used: 
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
	- unknowndropper
	- Volatile Cedar/Explosive
	
## Implementation and comparison of the following frameworks
#### Baseline Models:
- Bigram : Endgame’s Bigram model from dga_predict.
- LSTM : Endgame’s LSTM model from dga_predict.
- CNN : CNN adapted from Keegan Hine’s snowman.
- LSTM + CNN : CNN adapted from Keegan Hine’s snowman, combined 	with an LSTM as defined by Deep Learning For Realtime Malware Detection (ShmooCon 2018)’s LSTM + CNN (see 13:17 for architecture) by Domenic Puzio and Kate Highnam.

#### ALOHA Extended Models
##### (each simply use the 11 malware families as additional binary labels):
- ALOHA CNN
- ALOHA Bigram
- ALOHA LSTM
- ALOHA CNN+LSTM <br />
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

## References
1. [ALOHA: Auxiliary Loss Optimization for Hypothesis Augmentation](https://arxiv.org/abs/1903.05700)
2. [Covert.io](http://www.covert.io/auxiliary-loss-optimization-for-hypothesis-augmentation-for-dga-domain-detection/)
3. [DGA CapsNet: 1D Application of Capsule Networks to DGA Detection](https://www.mdpi.com/2078-2489/10/5/157/htm#B31-information-10-00157)
4. [CapsNet Implementation Keras](https://github.com/XifengGuo/CapsNet-Keras)
5. https://github.com/baderj/domain_generation_algorithms
6. [Deep Learning For Realtime Malware Detection (ShmooCon 2018)](https://www.youtube.com/watch?v=99hniQYB6VM) 
7. [Derived CNN model from Keegan Hines' Snowman Model](https://github.com/keeganhines/snowman/)
