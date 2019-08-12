# OCR assignment report
## Feature Extraction (Max 200 Words)
[I used 20 pca for reducing the dimension. I tried 40 principal components too, but it somehow seems to be better result to use only 20 pca. I calculated the eigenvector with the train_data and stored it in the dictionary to use it again for the test data. Among these 20 components I selected 10 features by adding up the divergence between each letters and get the best 10. However I excluded z and j and all the capital letters because of low samples that is not helpful for the divergence. Tried to include some symbols like (“” ‘  : , | .) But it didn’t lead to better performance so I excluded them again.]
## Classifier (Max 200 Words)
[I basically used nearest neighbor classification. I did the divergence step again only for train data to get 10 best features that is same as 10 best features for the test_data. By calculating, the dot product and outer product, I get the cosine distance and then with this distance I did classification. Page is basically test_data so I just changed the name to test for the convenience reason]
## Error Correction (Max 200 Words)
[I used the width of the letter to detect the start of the word and used the space between letters to detect the end of the letter. If the width is less than 4 then it is likely to be symbols like comma. If the space between letters is bigger than 10 or less than minus, it is the end of the word. When new line start, the new word begins. If the word is in the wordlist, I moved to the next word. Also, if the word starts with the capital then it will be people’s name, so I moved to the next word. If the word is not in the list, then I found the similar words from the wordlist that have the same length. I didn’t use error correction in my code because it does not lead to better performance. The reason is that after getting the list of all the similar words from the word list, I just took the first one and change the original word to this, because there is no way to find out the right among the list of similar words word unless there is program to understand the context. ]
## Performance
The percentage errors (to 1 decimal place) for the development data are
as follows:
‐ Page 1: [97.8%]
‐ Page 2: [97.8%]
‐ Page 3: [89.4%]
‐ Page 4: [73.4%]
‐ Page 5: [57.8%]
‐ Page 6: [42.8%]
## Other information (Optional, Max 100 words)
[I used first few samples for testing the error correction because it take much time (more than 2 minutes)
I used the median filter to reduce the noise level. It seems to slightly reduce the performance of page1 and page 2 (clear image) but give the worthy increase in the performance of page 3,4,5,6 (high noise image). Hence, I used it.
The commented code for 40 pca that I finally didn’t use contains divergence steps twice. It is because if I store all the 40 pac data in the model.json will be out of limit. I picked out 10 best features for the test and train data.
]