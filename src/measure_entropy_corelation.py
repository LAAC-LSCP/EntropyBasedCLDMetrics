import pandas as pd

def calculate_correlation(path):
    
    entropy_test = pd.read_csv(path)
    entropy_test = entropy_test[['entropy','gold_entropy']]
    entropy_gold = pd.read_csv('/scratch2/sdas/modules/EntropyBasedCLDMetrics/results/Thomas_30h_Librispeech360_en.csv')
    entropy_gold = entropy_gold[['entropy','gold_entropy']]
    correlation_matrix = entropy_gold.corrwith(entropy_test, method='pearson')
    
    print(correlation_matrix)
    
if __name__=='__main__':
    calculate_correlation('/scratch2/sdas/modules/EntropyBasedCLDMetrics-laac/results/cougar_thomas_30h_en.csv')