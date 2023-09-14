# image-captcha-solver

## 簡介
使用 python 訓練AI模型辨識圖片驗證碼上的文字，檔案包含三種不同的圖片驗證碼類型，
images 下的圖片檔為訓練使用之圖片，部分圖片已先做過預處理，圖片預處理的程式不包含在此。

## 如何使用
1. 請先自行上網下載 python，及安裝相關套件包，執行 python 程式，如有套件包缺失，可由提示文字上網查詢並下載安裝。  
   可使用以下語法執行程式
````
python train_image.py
python test_image.py
````
2. train_image.py 為訓練AI模型、test_image.py 為測試模型，如果想訓練新的圖片，可修改檔案參數後自行測試。
3. *.pth 為訓練完成的模型。

## 備註
此專案僅為個人記錄留存，不會進行維護。
