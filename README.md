# Word Embeddings 
The repository has five objectives
1. Frequency based embeddings 
2. Evaluating pretrained embeddings 
3. Aligning monolingual embeddings
4. Evaluating the bias in pretrained static embeddings using WEAT and WEFAT
5. Evaluating gender bias in contextual embeddings in encoder only and decoder only models



## 1. Frequency based embeddings 
To calculate co-occurrence matrices and PPMI, evaluation on Wordsim 353, differential drift: download the text corpus and run:
``` python cooccurrence_matrix/co-occ_matrix.py```

(Uncomment the commented part for the initial results as in the report)

To compute the pruned svd,ppmi, vocab and evalute on drit, wordsim run 
```python cooccurrence_matrix/svd_.py```

To evaluate on these on Google analogy test, Outliers test, WordSimilarity, run
```python cooccurrence_matrix/evaluate_metrics.py```

## 2. Evaluating pretrained embeddings 
To evaluate pretrained embeddings on same metrics, run
```python cooccurrence_matrix/evaluate_pretrained.py ```
this currently only supports fasttest wiki embeddings

## 3. Aligning monolingual embeddings
run 
``` python alignment.py```

Currently has :
1. Procrustes analysis
2. Generalized proscustes analysis
3. Fused gromov Wasserstein
4. Bisparse alignment (broken- does not converge)

## 4. Evaluating the bias in pretrained static embeddings using WEAT and WEFAT
Run 
```python we(f)at_en.py```

For hindi run
```python weat-hi.py```


## 5. Evaluating gender bias in contextual embeddings in encoder only and decoder only models

Suggested to run these on colab/ kaggle on gpu as loading the models could be heavy

```gender_bias_encoder.ipynb```
```gender-bias-gemma2-2b-it.ipynb```

For encoder, please do restart the session after pip uninstall numpy , installing the correct version to avoid dependency issues with transformer-lens HookedEncoder


For set up

### Option 1: Pip (Recommended for most users)
```bash
git clone https://github.com/ilatims-b/monolingual_embedding_alignment.git
cd monolingual_embedding_alignment
python -m venv alignvec
source alignvec/bin/activate  # On Windows: alignvec\Scripts\activate
pip install -r requirements.txt
```
### Option 2:Conda (Full reproducibility, includes MPS/CUDA support)
```bash
conda env create -f environment.yml
conda activate alignvec
```



