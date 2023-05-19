# GenesPaperPseudocode
Paper title: "A Machine Learning Approach to Parkinson’s Disease Blood Transcriptomics"

This is a translation of the following pseudocode provided in the paper. 

Appendix A
Algorithm A1 Pseudocode.

Let F be the total number of features
forr = 1 to 20 do
Divide data into 10 stratified folds using random seed r
     for fold k = 1 to 10 do
         Set fold k as validation_set and the remaining 9 folds as training_set
         for s = 1 to 100 do 
              Divide training_set into 5 stratified folds using random seed s
              Take 4 of the folds as the new training set
              Train a RF on this training set with 1000 trees
            for f = 1 to F do
                    Set is_outlierr,s,f = 0
                    Estimate importancer,s,f
            end for
            Evaluate thr,s = MEDIAN(f)
(importancer,s,f) + 1.5 ∗ IQR(f)
(importancer,s,f) where MEDIAN(f)
 means median over the F values of f
            for f = 1 to F do
                    is_outlierr,s,f = IFELSE(importancer,s,f> thr,s, 1, 0)
            end for
         end for
         for f = 1 to F do
             Set percentage_outlierr,f= 0
             for s = 1 to 100 do
                    percentage_outlierr,f
 += is_outlierr,s,f
             end for
         end for
         for C = 1 to 100 do
             Evaluate is_selectedr,f,C = IFELSE(percentage_outlierr,f> C, 1, 0)
             Train XGBoost on the training_set
 using only features f with is_selectedr,f,C =128:
             Estimate performance ROCAUCr,k,C
 on the validation_set
         end for
     end for
end for
forC = 1 to 100 do
     Evaluate m_ROCAUCr,C
 = MEDIAN(k)
(ROCAUCr,k,C
) over the 10 values of k
end for
forC = 1 to 100 do
     Evaluate m_ROCAUCC
 = MEDIAN(r)
(ROCAUCr,C
) over the 20 values of r
end for
Let C∗
 = ARGMAXC
(m_ROCAUCC
)
forf = 1 to F do
     Set count_selectedf
 = 0
     for r = 1 to 20 do
         count_selectedf
 += is_selectedr,f,C∗
     end for
end for
