本文档主要用于说明配置文件中各项的含义，根据不同的需求可以扩展和更改。目前各参数所填的为默认值。

- **bconfig.json**

  ```json
     参数名称                           		参数说明                     
  "data": 
      "separator": "^^",              	//文本分隔符
      "columnNum": 3,                 	//文本列数
      "group": "jsonPath",            	//按group对应列的值分组
      "timeColumn": "timestamp",      	//时间戳列名
      "countColumn": "occurNum",      	//出现次数列名
      "columns": [                    	//所有输入列名，与文本列名一致
          "jsonPath",
          "occurNum",
          "timestamp"
       ]
  "vecModel": "wordModel_50_4.model", 	//存储训练完Word2Vec模型的位置
  "windowSize": 14                    	//滑动窗口大小（时序特征大小）
  ```

  

- **config.json**

  ```json
     参数名称                                               参数说明
  "data":                                                //*数据集相关属性
  	"trainName": "data/14_feature_train.txt",          //训练数据位置 
  	"predictName": "data/14_feature_predict.txt",      //预测数据位置
  	"separator": "^^",                                 //文本分隔符
  	"columnNum": 3,                                    //文本列数
  	"nonSeqFeatureSize": 53,                           //非时序特征大小
  											（JSONPath, Path特征大小 + 时间特征大小）
  	"group": "jsonPath",							   //按group对应列的值分组
  	"timeColumn": "timestamp",						   //时间戳列名
  	"countColumn": "occurNum",						   //出现次数列名
      "columns": [                                       //所有输入列名，与文本列名一致
          "jsonPath",
          "occurNum",
          "timestamp"
      ]
  "windowSize": 14,									   //滑动窗口大小（时序特征大小）
  "windowsNum": 50,									   //有多少窗口数
  											（最多窗口数，数据量小时会取计算出的大小）
  "tagsetSize": 4									       //crf层状态大小（B,0,1,E）
  
  "training":                                            //*训练模型相关参数
      "epoch": 2,                                        //迭代训练次数
      "batchSize": 100,                                  //每个batch有多少个样本
      "shuffle": true,                                   //每个epoch开始的时候，数据是否打乱
      "learningRate": 0.001                              //学习率（经验值）
  
  "model": 											  //*模型相关参数
      "lstm_crf": 
          "inputSize": 81,                              //输入x的特征维度
          "hiddenSize": 20,                             //隐藏层的特征维度
          "numLayers": 2,                               //LSTM隐藏层数
          "batchFirst": true,                       //真则输入输出格式为(batch,seq,feature)
          "dropout": 0.5                                //损失率（避免过拟合）
  
      "linear1":                             		      //*简单的全连接，用于将lstm层降维
          "inFeatures": 20,                             //同hiddenSize大小
          "outFeatures": 4                              //tagsetSize
  	"saveName": "model/lstmcrf_14.pt"                 //模型保存路径
  
  "prediction":                                         //*预测相关参数          
      "output": "data/prediction/predict_result_lstmcrf_14.txt"  //预测数据文件保存路径
  
  
  ```
  
  

