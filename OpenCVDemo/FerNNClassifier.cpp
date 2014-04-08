/*
 * FerNNClassifier.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 */

#include "FerNNClassifier.h"

using namespace cv;
using namespace std;

void FerNNClassifier::read(const FileNode& file){
  ///Classifier Parameters
  valid = (float)file["valid"];
  ncc_thesame = (float)file["ncc_thesame"];
  nstructs = (int)file["num_trees"];
  structSize = (int)file["num_features"];
  thr_fern = (float)file["thr_fern"];
  thr_nn = (float)file["thr_nn"];
  thr_nn_valid = (float)file["thr_nn_valid"];
  nExLimit=(int)file["nExLimit"];
  pExLimit=(int)file["pExLimit"];
  nExTimes=0;
  pExTimes=0;
  nExInfo=new ExInfo[nExLimit];
  pExInfo=new ExInfo[pExLimit];
}

void FerNNClassifier::prepare(const vector<Size>& scales){
  acum = 0;
  //Initialize test locations for features
  int totalFeatures = nstructs*structSize;
  features = vector<vector<Feature> >(scales.size(),vector<Feature> (totalFeatures));
  RNG& rng = theRNG();
  float x1f,x2f,y1f,y2f;
  int x1, x2, y1, y2;
  for (int i=0;i<totalFeatures;i++){
      x1f = (float)rng;
      y1f = (float)rng;
      x2f = (float)rng;
      y2f = (float)rng;
      for (int s=0;s<scales.size();s++){
          x1 = x1f * scales[s].width;
		  //乘以宽度后得到了第几列，所以在ferNNclassifier中用了patch.at<uchar>(y1,x1) > patch.at<uchar>(y2, x2);
		  //important,patch.at<uchar>(y1,x1)得到的是第y1行，第x1列
          y1 = y1f * scales[s].height;//第y1行
          x2 = x2f * scales[s].width;//
          y2 = y2f * scales[s].height;
          features[s][i] = Feature(x1, y1, x2, y2);
      }
  }
  //Thresholds
  thrN = 0.5*nstructs;

  //Initialize Posteriors
  for (int i = 0; i<nstructs; i++) {
      posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
      pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
      nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
  }
}

void FerNNClassifier::getFeatures(const cv::Mat& image,const int& scale_idx, vector<int>& fern){
  int leaf;
  for (int t=0;t<nstructs;t++){
      leaf=0;
      for (int f=0; f<structSize; f++){
          leaf = (leaf << 1) + features[scale_idx][t*structSize+f](image);
      }
      fern[t]=leaf;
  }
}

float FerNNClassifier::measure_forest(vector<int> fern) {
  float votes = 0;
  for (int i = 0; i < nstructs; i++) {
      votes += posteriors[i][fern[i]];
  }
  return votes;
}
void updatePoscu(float *upPos,int *unPosInd);
void FerNNClassifier::update(const vector<int>& fern, int C, int N) {
  int idx;
 // float *upPos=new float[nstructs*2];//前nstructs格式fern值，后面试posteriors值
  float *upPos=new float[nstructs];
  int *upPosInd=new int [nstructs];
  for (int i = 0; i < nstructs; i++) {
      idx = fern[i];
      (C==1) ? pCounter[i][idx] += N : nCounter[i][idx] += N;
      if (pCounter[i][idx]==0) {
          posteriors[i][idx] = 0;
		 // upPos[i]=idx;		 
		 // upPos[i+nstructs]=0;
		   upPosInd[i]=idx;
		    upPos[i]=0;
		  //这里也有个问题是在tld.init时，这个数据就会更新，而那时候我们的gpu还没有任何数据啊
		  //如果把posteriors的大小设定放在tld.init前，那时候又无法知道nstructs的大小.
		  //好，tld的nstructs参数从文件读入的时间好像是在构造函数里，读完我们就往gpu送，怎么样
		  //也就是说把那些个参数设置放入构造函数。
      } else {
          posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
		//  upPos[i]=idx;
		//  upPos[i+nstructs]=posteriors[i][idx];
		   upPosInd[i]=idx;
		    upPos[i]=posteriors[i][idx];
      }
  }
   updatePoscu(upPos,upPosInd);//nstructs一次调用gpu的kernel函数，如果每更新一次就调用一次，这样效率不行
   delete[] upPos;
   delete[] upPosInd;
}

void FerNNClassifier::trainF(const vector<std::pair<vector<int>,int> >& ferns,int resample){
  // Conf = function(2,X,Y,Margin,Bootstrap,Idx)
  //                 0 1 2 3      4         5
  //  double *X     = mxGetPr(prhs[1]); -> ferns[i].first
  //  int numX      = mxGetN(prhs[1]);  -> ferns.size()
  //  double *Y     = mxGetPr(prhs[2]); ->ferns[i].second
  //  double thrP   = *mxGetPr(prhs[3]) * nTREES; ->threshold*nstructs
  //  int bootstrap = (int) *mxGetPr(prhs[4]); ->resample
  float temp=0;
  thrP = thr_fern*nstructs;                                                          // int step = numX / 10;
  //for (int j = 0; j < resample; j++) {                      // for (int j = 0; j < bootstrap; j++) {
      for (int i = 0; i < ferns.size(); i++){               //   for (int i = 0; i < step; i++) {
                                                            //     for (int k = 0; k < 10; k++) {
                                                            //       int I = k*step + i;//box index
                                                            //       double *x = X+nTREES*I; //tree index
          if(ferns[i].second==1){ //1代表是good_box                          //       if (Y[I] == 1) {
              if(measure_forest(ferns[i].first)<=thrP)      //         if (measure_forest(x) <= thrP)
                update(ferns[i].first,1,1);                 //             update(x,1,1);
          }else{                                            //        }else{
              if ((temp=measure_forest(ferns[i].first)) >= thrN){   //         if (measure_forest(x) >= thrN)
				//measure_forest(ferns[i].first);
				  update(ferns[i].first,0,1);   }              //             update(x,0,1);
          }
      }
  //}
}
void FerNNClassifier::trainNN(const vector<cv::Mat>& nn_examples){
  float conf,dummy;
  vector<int> y(nn_examples.size(),0);
  y[0]=1;
  vector<int> isin;
  for (int i=0;i<nn_examples.size();i++){                          //  For each example
      NNConf(nn_examples[i],isin,conf,dummy);                      //  Measure Relative similarity
      if (y[i]==1 && conf<=thr_nn){                                //    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65
          if (isin[1]<0){                                          //      if isnan(isin(2))
             //1 pEx = vector<Mat>(1,nn_examples[i]);                 //        tld.pex = x(:,i);
			 //1 addpExgpu(nn_examples[i]);
			  addpEx(nn_examples[i],conf,pExTimes);
		  pExTimes++;
              continue;                                            //        continue;
          }                                                        //      end
          //////////////pEx.insert(pEx.begin()+isin[1],nn_examples[i]);        //      tld.pex = [tld.pex(:,1:isin(2)) x(:,i) tld.pex(:,isin(2)+1:end)]; % add to model
		  //pEx.push_back(nn_examples[i]);
		  //addpExgpu(nn_examples[i]);
		  addpEx(nn_examples[i],conf,pExTimes);
		  pExTimes++;
      }                                                            //    end
      if(y[i]==0 && conf>0.5)                                      //  if y(i) == 0 && conf1 > 0.5
	  {		  
		  addnEx(nn_examples[i],conf,nExTimes);
		  nExTimes++;
	//1	nEx.push_back(nn_examples[i]);                             //    tld.nex = [tld.nex x(:,i)];
	//1    addnExgpu(nn_examples[i]);
	  }
  }                                                                 //  end
  acum++;
  printf("%d. Trained NN examples: %d positive %d negative\n",acum,(int)pEx.size(),(int)nEx.size());
}                                                                  //  end

void opcvNcc_CCORR_NORMED(cv::Mat &src,const cv:: Mat &dst,float &ncc){	
	Mat ncctmp(1,1,CV_32F);
	matchTemplate(src,dst,ncctmp,CV_TM_CCORR_NORMED);     //measure NCC to negative examples
	ncc=((float*)ncctmp.data)[0];
}

void myNcc_CCORR_NORMED(cv::Mat &src,const cv:: Mat &dst,float &ncc){	//这是归一化相关匹配法，method=CV_TM_CCORR_NORMED，自动判断是8U还是32F
	if(src.step[1]==1){//实际是指src.step.buf的两个值，第一个src.step[0]是矩阵的行宽（单位是字节），第二个是矩阵的数据类型的大小（字节单位）
		double srcCFH=0.0,dstCFH=0.0,nccSum=0.0;
		for (int i=0;i<src.rows;i++)
			for (int j=0;j<src.cols;j++)
			{
				srcCFH+=(src.at<uchar>(i,j))*(src.at<uchar>(i,j));
				dstCFH+=(dst.at<uchar>(i,j))*(dst.at<uchar>(i,j));
				nccSum+=(src.at<uchar>(i,j))*(dst.at<uchar>(i,j));
			}
			double CFH=sqrt((double)srcCFH)*sqrt((double)dstCFH);
			ncc=(double)nccSum/CFH;
	}
	else{

		double srcCFH=0.0,dstCFH=0.0,nccSum=0.0;
		for (int i=0;i<src.rows;i++)
			for (int j=0;j<src.cols;j++)
			{
				srcCFH+=(src.at<float>(i,j))*(src.at<float>(i,j));
				dstCFH+=(dst.at<float>(i,j))*(dst.at<float>(i,j));
				nccSum+=(src.at<float>(i,j))*(dst.at<float>(i,j));
			}
			double CFH=sqrt((double)srcCFH)*sqrt((double)dstCFH);
			ncc=(double)nccSum/CFH;
	}


}

void myNcc1(cv::Mat &src,cv:: Mat &dst,float &ncc){//这是8U的归一化相关系数匹配法，method=CV_TM_CCOEFF_NORMED
	int srcSum=0,dstSum=0;
	int row=src.rows;
	int col=src.cols;
	for (int i=0;i<src.rows;i++)
		for (int j=0;j<src.cols;j++)
		{
			srcSum+=src.at<uchar>(i,j);
			dstSum+=dst.at<uchar>(i,j);
		}
		double srcAver=(double)srcSum/(src.rows*src.cols);
		double dstAver=(double)dstSum/(dst.rows*dst.cols);
		int srcCFH=0.0,dstCFH=0.0,nccSum=0.0;
		for (int i=0;i<src.rows;i++)
			for (int j=0;j<src.cols;j++)
			{
				srcCFH+=(src.at<uchar>(i,j)-srcAver)*(src.at<uchar>(i,j)-srcAver);
				dstCFH+=(dst.at<uchar>(i,j)-dstAver)*(dst.at<uchar>(i,j)-dstAver);
				nccSum+=(src.at<uchar>(i,j)-srcAver)*(dst.at<uchar>(i,j)-dstAver);
			}
			double CFH=sqrt((double)srcCFH)*sqrt((double)dstCFH);
			ncc=(double)nccSum/CFH;
			if(ncc<0) ncc=ncc*(-1);
}

void FerNNClassifier::NNConf(const Mat& example, vector<int>& isin,float& rsconf,float& csconf){
  /*Inputs:
   * -NN Patch
   * Outputs:
   * -Relative Similarity (rsconf), Conservative Similarity (csconf), In pos. set|Id pos set|In neg. set (isin)
   */
  isin=vector<int>(3,-1);
  if (pEx.empty()){ //if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
      rsconf = 0; //    conf1 = zeros(1,size(x,2));
      csconf=0;
      return;
  }
  if (nEx.empty()){ //if isempty(tld.nex) % IF negative examples in the model are not defined THEN everything is positive
      rsconf = 1;   //    conf1 = ones(1,size(x,2));
      csconf=1;
      return;
  }
  Mat ncc(1,1,CV_32F);
  float nccP,csmaxP=0,maxP=0,myncc=0;
  bool anyP=false;
  int maxPidx=0,validatedPart = ceil(pEx.size()*valid);
  float nccN, maxN=0;
  bool anyN=false;
//   gpu::GpuMat gpu_scene, gpu_templ,gpu_result;	
//   	vector<Mat> rgbImg(3);	
//	vector<Mat> rgbImg2(3);
	Mat pExi,examplei,nExi;
//	example.convertTo(examplei,CV_8U);
//	gpu_templ.upload(examplei);
	//gpu_templ.upload(examplei);
  for (int i=0;i<pEx.size();i++){

	//要使用gpu，首先要把图像从32f转为8u.
  
    //cvtColor(pEx[i],pExi,CV_RGB2GRAY);
    //cvtColor(example,examplei,CV_RGB2GRAY);
	//split(pEx[i], rgbImg);
	//Mat scene1chan=rgbImg[0];	
	//split(example, rgbImg2);
	//Mat templ1chan=rgbImg2[0];
	//gpu_scene.upload(scene1chan);
	//gpu_templ.upload(templ1chan);
//	pEx[i].convertTo(pExi,CV_8U);	
//	gpu_scene.upload(pExi);	
//	gpu_result.create(gpu_scene.rows-gpu_templ.rows+1 , gpu_scene.cols-gpu_templ.cols+1 , CV_32F);
//	gpu::matchTemplate(gpu_scene, gpu_templ, gpu_result,CV_TM_CCORR_NORMED);      // measure NCC to positive examples	  
//	ncc=Mat(gpu_result);
//	matchTemplate(pEx[i],example,ncc,CV_TM_CCORR_NORMED);
//	matchTemplate(pExi,examplei,ncc,CV_TM_CCORR_NORMED);
    //normalize( ncc, ncc, 0, 1, NORM_MINMAX, -1, Mat() );
//float	nccP1=(((float*)ncc.data)[0]+1)*0.5;//这么做的原因是CV_TM_CCORR_NORMED,归一化（标准）相关匹配CV_TM_CCOEFF 相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。
	 
	  
	  //myNcc_CCORR_NORMED(pEx[i],example,myncc);//使用自己编写的代码
	  opcvNcc_CCORR_NORMED(pEx[i],example,myncc);//使用opcv的代码

	  nccP=(myncc+1)*0.5;
//	if(nccP1-nccP>0.0001||nccP1-nccP<-0.0001)
//	{
//		printf("p=%f,p1=%f",nccP,nccP1);system("PAUSE");
//	}
      if (nccP>ncc_thesame)
        anyP=true;
      if(nccP > maxP){
          maxP=nccP;
          maxPidx = i;
          if(i<validatedPart)
            csmaxP=maxP;
      }
  }

  for (int i=0;i<nEx.size();i++){
	  //myNcc_CCORR_NORMED(nEx[i],example,myncc);
	  opcvNcc_CCORR_NORMED(nEx[i],example,myncc);

	  nccN=(myncc+1)*0.5;
      if (nccN>ncc_thesame)
        anyN=true;
      if(nccN > maxN)
        maxN=nccN;
  }

  //set isin
  if (anyP) isin[0]=1;  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
  isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
  if (anyN) isin[2]=1;  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them
  //Measure Relative Similarity
  float dN=1-maxN;
  float dP=1-maxP;
  rsconf = (float)dN/(dN+dP);
  //Measure Conservative Similarity
  dP = 1 - csmaxP;
  csconf =(float)dN / (dN + dP);
}

void FerNNClassifier::evaluateTh(const vector<pair<vector<int>,int> >& nXT,const vector<cv::Mat>& nExT){
float fconf;
  for (int i=0;i<nXT.size();i++){
    fconf = (float) measure_forest(nXT[i].first)/nstructs;
    if (fconf>thr_fern)
      thr_fern=fconf;
}
  vector <int> isin;
  float conf,dummy;
  for (int i=0;i<nExT.size();i++){
      NNConf(nExT[i],isin,conf,dummy);
      if (conf>thr_nn)
        thr_nn=conf;
  }
  if (thr_nn>thr_nn_valid)
    thr_nn_valid = thr_nn;
}

void FerNNClassifier::show(){
  Mat examples((int)pEx.size()*pEx[0].rows,pEx[0].cols,CV_8U);
  double minval;
  Mat ex(pEx[0].rows,pEx[0].cols,pEx[0].type());
  for (int i=0;i<pEx.size();i++){
    minMaxLoc(pEx[i],&minval);
    pEx[i].copyTo(ex);
    ex = ex-minval;
    Mat tmp = examples.rowRange(Range(i*pEx[i].rows,(i+1)*pEx[i].rows));
    ex.convertTo(tmp,CV_8U);
	// imshow("tmp",tmp);
  }
 // printf("pEx NUMBER is: %d\n", pEx.size());
  imshow("Examples",examples);
}
void addpExcu(const float *spEx);
void addnExcu1(const float *snEx);
void addnExcu(const float *snEx,int position,bool full);
void addpExcu(const float *spEx,int position,bool full);
void NNConfcu(const float *sexample,float *smatch);
void NNConfBatchcu(void ** batch,float *nccBatAnscu,int count,double &f2datacost);
void testDatacu(const float *in,float *pout,float *nout);
//输入的是一个指针数组这个指针数组大小detections行，1列。输出的是结果数据存入nccBatAns,这是个有detections行pEx.size()+nEx.size()列的二维数组
int FerNNClassifier:: getnExPosition()
{
    //把最旧的一半相似度最小的删了	
    //length=sizeof(nExInfo)/sizeof(ExInfo);

	int *ban=new int[nExLimit/2];

	for (int i=0;i<nExLimit;i++)
	{
		if (i<nExLimit/2)
		{
			ban[i]=i;continue;
		}
		int maxt=0;
		for (int j=1;j<nExLimit/2;j++)
		{
			if(nExInfo[ban[j]].times>nExInfo[ban[maxt]].times)
				maxt=j;
		}//newt代表ban里nExInfo[ban[newt]]最大的
		//如果下面遇到比这个最大的小的，则替换
		if (nExInfo[i].times<nExInfo[ban[maxt]].times)
			ban[maxt]=i;


	}
	//最终ban里存的是times最小的一半nExInfo的下标，然后接下来选一个conf最小的下标出来
	int minn=0;//我们要返回时间最小的下标
	for(int i=1;i<nExLimit/2;i++)
	{
		
		if(nExInfo[ban[i]].conf<nExInfo[ban[minn]].conf)
			minn=i;
	}
	return ban[minn];

}

int FerNNClassifier:: getpExPosition()
{
    //把最旧的一半相似度最大的删了	
    //length=sizeof(pExInfo)/sizeof(ExInfo);

	int *ban=new int[pExLimit/2];

	for (int i=0;i<pExLimit;i++)
	{
		if (i<pExLimit/2)
		{
			ban[i]=i;continue;
		}
		int maxt=0;
		for (int j=1;j<pExLimit/2;j++)
		{
			if(pExInfo[ban[j]].times>pExInfo[ban[maxt]].times)
				maxt=j;
		}//newt代表ban里pExInfo[ban[newt]]最大的
		//如果下面遇到比这个最大的小的，则替换
		if (pExInfo[i].times<pExInfo[ban[maxt]].times)
			ban[maxt]=i;


	}
	//最终ban里存的是times最小的一半nExInfo的下标，然后接下来选一个conf最大的下标出来
	int maxn=0;//我们要返回时间最小的下标
	for(int i=1;i<pExLimit/2;i++)
	{
		
		if(pExInfo[ban[i]].conf>pExInfo[ban[maxn]].conf)
			maxn=i;
	}
	return ban[maxn];

}


//这个函数的功能是，输入一个nex的mat,conf,第几个，加入到nEx中
void FerNNClassifier::addnEx(const cv::Mat &n,float conf,int times)
{//把最旧的一半相似度最大的删了
	//先是cpu
	int nExPosition;

	ExInfo exTemp={ times,conf};
	if(nEx.size()<nExLimit)//如果nEx的大小小于上限，则直接加入到后面，gpu上也是
	{
		nEx.push_back(n);
		addnExcu(&n.ptr<float>(0)[0],0,false);
		nExInfo[nEx.size()-1]=exTemp;
	}
	else
	{//否则找到相关地方替换
		nExPosition=getnExPosition();
		nEx[nExPosition]=n;
		addnExcu(&n.ptr<float>(0)[0],nExPosition,true);
		nExInfo[nExPosition]=exTemp;
	}
}


void FerNNClassifier::addpEx(const cv::Mat &n,float conf,int times)
{//把最旧的一半相似度最小的删了
	//先是cpu
	int pExPosition;

	ExInfo exTemp={ times,conf};
	if(pEx.size()<pExLimit)//如果pEx的大小小于上限，则直接加入到后面，gpu上也是
	{
		pEx.push_back(n);
		addpExcu(&n.ptr<float>(0)[0],0,false);
		pExInfo[pEx.size()-1]=exTemp;
	}
	else
	{//否则找到相关地方替换
		pExPosition=getpExPosition();
		pEx[pExPosition]=n;
		addpExcu(&n.ptr<float>(0)[0],pExPosition,true);
		pExInfo[pExPosition]=exTemp;
	}
}


void FerNNClassifier::addpExgpu(const cv::Mat &p)
{
	addpExcu(&p.ptr<float>(0)[0]);
}
void FerNNClassifier::addnExgpu(const cv::Mat &n)
{
	addnExcu1(&n.ptr<float>(0)[0]);
	
}

void FerNNClassifier::NNConfBatchgpu(void **batch, float *nccBatAns ,int detections,double &f2datacost)
{//输入的是一个指针数组这个指针数组大小detections行，1列，每个元素指向一个地址，该地址是一个cv::Mat的数据地址，要
//输出的是的数据存入nccBatAns,这是个有detections行(pEx.size()+nEx.size())列的二维数组。
	float * nccBatAnscu=new float[(pEx.size()+nEx.size())*detections];
	NNConfBatchcu(batch,nccBatAnscu,detections,f2datacost);

	for(int d=0;d<detections;d++){
	//变量照抄
	float *smatch=nccBatAnscu+d*(pEx.size()+nEx.size());
	float rsconf,csconf;
	int isin[3]={-1,-1,-1};
	  float nccP,csmaxP=0,maxP=0;
  bool anyP=false;
  int maxPidx=0,validatedPart = ceil(pEx.size()*valid);//让ferNNClassifier中也有pEx和nEx
  float nccN, maxN=0;
  bool anyN=false;
  //依次处理nccBatAnscu的每一行，然后输出到nccBatAns
	  for (int i=0;i<pEx.size();i++){
        if (smatch[i]>ncc_thesame)
        anyP=true;
      if(smatch[i] > maxP){
          maxP=smatch[i];
          maxPidx = i;
          if(i<validatedPart)
            csmaxP=maxP;
      }
  }
  for(int j=pEx.size();j<pEx.size()+nEx.size();j++)
  {
	       if (smatch[j]>ncc_thesame)
        anyN=true;
      if(smatch[j] > maxN)
        maxN=smatch[j];
  }

  //set isin
  if (anyP) isin[0]=1;  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
  isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
  if (anyN) isin[2]=1;  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them
  //Measure Relative Similarity
  float dN=1-maxN;
  float dP=1-maxP;
  rsconf = (float)dN/(dN+dP);
  //Measure Conservative Similarity
  dP = 1 - csmaxP;
  csconf =(float)dN / (dN + dP);

 /////////////////////////////////上面完了，然后赋值给nccBatAns结果的每一行
    nccBatAns[d*5]=rsconf;
  nccBatAns[d*5+1]=csconf;
  nccBatAns[d*5+2]=isin[0];
  nccBatAns[d*5+3]=isin[1];
  nccBatAns[d*5+4]=isin[2];
  ///////////

	}
	delete[] nccBatAnscu;
}
void FerNNClassifier::NNConfgpu(const Mat& example, vector<int>& isin,float& rsconf,float& csconf)
{
	

	 /*Inputs:
   * -NN Patch
   * Outputs:
   * -Relative Similarity (rsconf), Conservative Similarity (csconf), In pos. set|Id pos set|In neg. set (isin)
   */
  isin=vector<int>(3,-1);
  if (pEx.empty()){ //if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
      rsconf = 0; //    conf1 = zeros(1,size(x,2));
      csconf=0;
      return;
  }
  if (nEx.empty()){ //if isempty(tld.nex) % IF negative examples in the model are not defined THEN everything is positive
      rsconf = 1;   //    conf1 = ones(1,size(x,2));
      csconf=1;
      return;
  }
  Mat ncc(1,1,CV_32F);
  float nccP,csmaxP=0,maxP=0;
  bool anyP=false;
  int maxPidx=0,validatedPart = ceil(pEx.size()*valid);//让ferNNClassifier中也有pEx和nEx
  float nccN, maxN=0;
  bool anyN=false;
  ///////////////////////////////前面都不变，下面这两个循环一快让gpu来做///////////////////////////////////////////////
  float *smatch=new float[pEx.size()+nEx.size()];
 // float szl[60][15];
 // float szl1[60][15];
 // float szp[15][15];
 // float szp1[15][15];
 // float ans[100];
//  testDatacu(&pEx[0].ptr<float>(0)[0],(float*)szp1,(float*)szl1);
 //  memcpy((void*)szp,(void*)&pEx[0].ptr<float>(0)[0],15*15*4);
//  for(int i=0;i<4;i++)
 // memcpy((void*)&szl[15*i][0],(void*)&nEx[i].ptr<float>(0)[0],15*15*4);

  NNConfcu(&example.ptr<float>(0)[0],smatch);//example 是输入，smatch是输出
 // memcpy((void*) ans,smatch,(pEx.size()+nEx.size())*4);

  /////////////////////////////gpu返回结果//////////////////////////////////////////////////////////
 /*
  for (int i=0;i<pEx.size();i++){
      matchTemplate(pEx[i],example,ncc,CV_TM_CCORR_NORMED);      // measure NCC to positive examples
      nccP=(((float*)ncc.data)[0]+1)*0.5;
      if (nccP>ncc_thesame)
        anyP=true;
      if(nccP > maxP){
          maxP=nccP;
          maxPidx = i;
          if(i<validatedPart)
            csmaxP=maxP;
      }
  }
  for (int i=0;i<nEx.size();i++){
      matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);     //measure NCC to negative examples
      nccN=(((float*)ncc.data)[0]+1)*0.5;
      if (nccN>ncc_thesame)
        anyN=true;
      if(nccN > maxN)
        maxN=nccN;
  }
  */
  for (int i=0;i<pEx.size();i++){
        if (smatch[i]>ncc_thesame)
        anyP=true;
      if(smatch[i] > maxP){
          maxP=smatch[i];
          maxPidx = i;
          if(i<validatedPart)
            csmaxP=maxP;
      }
  }
  for(int j=pEx.size();j<pEx.size()+nEx.size();j++)
  {
	        if (smatch[j]>ncc_thesame)
        anyN=true;
      if(smatch[j] > maxN)
        maxN=smatch[j];
  }

  //set isin
  if (anyP) isin[0]=1;  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
  isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
  if (anyN) isin[2]=1;  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them
  //Measure Relative Similarity
  float dN=1-maxN;
  float dP=1-maxP;
  rsconf = (float)dN/(dN+dP);
  //Measure Conservative Similarity
  dP = 1 - csmaxP;
  csconf =(float)dN / (dN + dP);
  delete [] smatch;//用完以后释放掉内存
}