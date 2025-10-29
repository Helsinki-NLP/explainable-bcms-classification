# Explainable Linguistic Variety Classification: the Case of Bosnian-Croatian-Montenegrin-Serbian

Data and code for linguistic variety classification of Bosnian-Croatian-Montenegrin-Serbian with extraction of relevant linguistic features for each variety.

## Outline
Data, code and results in this repository rely on the following methodology:
- Train a classifier on the available corpus
- Use the classifier to extract most relevant linguistic features (i.e. words) for each class through the Leave-one-out approach (Xie et al. 2024).

Details are available in the following paper:

Aleksandra Miletić, Timothee Mickus, Yves Scherrer. (in press) Explainable Linguistic Variety Classification: the Case of Bosnian-Croatian-Montenegrin-Serbian. First Conference of the Regional Language Data Initiative (ReLDI). September 2025, Belgrade, Serbia.

## Content

### data
The dataset used here was collected from the social media platform Twitter (rebranded as X in 2023) (Rupnik et al. 2023). The original annotation was done by a single annotator, and in a single-label setup. We use a revised version of the dataset, in which the dev and test sets were reannotated in a multi-label setting, by multiple   annotators (Miletić and Miletić 2024). The largest class (Serbian) was downsized to the second-largest (Croatian). The original instances (full twitter productions of a single user) were split into sentence-based instances.

File list:
- twitter_dssr_train_sl.tsv: train split, single label annotation.
- twitter_dssr_dev_sl.tsv: dev split, single label annotation. Used for classifier training to simplify the training process. An original multilabel instance is inserted multiple times in the split as a single label instance with all of the labels from the original multilabel instance (one instance with the label 'bs,sr' becomes one instance with the label 'bs' and another with the label 'sr')
- twitter_dssr_dev_ml.tsv: dev split, multilabel annotation, used for evaluation and feature extraction.
- twitter_dssr_test_sl.tsv: test split, single label annotation.
- twitter_dssr_test_ml.tsv: test split, multilabel annotation, used for evaluation.

### scripts
- train.py: train classifier 
- eval_multilabel: evaluate on multilabel datasets. A prediction will be evaluated as correct if the predicted label appears in the label set in the gold annotation.
- loo_correct.py: extract relevant linguistic features from the classifier using the Leave-one-out method. Generates instance-level classification explanations.
- filter_sort_explanations.py: aggregate explanations across dataset and output 100 explanations per class.
- logistic_regression_ml.py: train and evaluate a logistic regression model to serve as a baseline. Able to deal with multilabel annotation for evaluation.

### models
- model_twitter_dssr_bertic: fine-tuned bertic-based classifier used in Miletić et al. (2025).
- model_twitter_dssr_xlmr: fine-tuned xlmr-based classifier used in Miletić et al. (2025).

### results
- logistic_regression: linguistic features extracted for each class from the logistic regression baseline for test and train splits.
- model_twitter_dssr_bertic: 
    - expl_correct.json: instance-level explanations produced by loo_correct.py
    - filtered_words_mc.txt: aggregated explanations across the test set; allows for explanations that appear for multiple classes (but not all of them).
    - filtered_words_sc.txt: aggregated explanations across the test set; allows only for explanations that appear for a single class.
- model_twitter_dssr_xlmr:
    - expl_correct.json: instance-level explanations produced by loo_correct.py
    - filtered_words_mc.txt: aggregated explanations across the test set; allows for explanations that appear for multiple classes (but not all of them).
    - filtered_words_sc.txt: aggregated explanations across the test set; allows only for explanations that appear for a single class.


## References
Aleksandra Miletić and Filip Miletić. 2024. A Gold Standard with Silver Linings: Scaling Up Annotation for Distinguishing Bosnian, Croatian, Montenegrin and Serbian. In Proceedings of the Fourth Workshop on Human Evaluation of NLP Systems (HumEval) @ LREC-COLING 2024, pages 36–46, Torino, Italia. ELRA and ICCL.

Peter Rupnik, Taja Kuzman, and Nikola Ljubešić. 2023. BENCHić-lang: A Benchmark for Discriminating between Bosnian, Croatian, Montenegrin and Serbian. In Tenth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial 2023), pages 113–120, Dubrovnik, Croatia. Association for Computational Linguistics.

Roy Xie, Orevaoghene Ahia, Yulia Tsvetkov, and Antonios Anastasopoulos. 2024. Extracting Lexical Features from Dialects via Interpretable Dialect Classifiers. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 54–69, Mexico City, Mexico. Association for Computational Linguistics.
