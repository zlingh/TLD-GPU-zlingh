#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_texture_types.h>////
#include<device_launch_parameters.h>///////我加的
#include<channel_descriptor.h>//
#include<texture_fetch_functions.h>//
#include<cuda_runtime.h>
#include<driver_types.h>
#include<cutil.h>
using namespace cv;
using namespace std;
#define ele2D(BaseAddress, x, y,pitch) *((float*)((char*)(BaseAddress) + (y) * (pitch)) + (x));
texture<unsigned char, 2> imageData2D;//当前帧的图像，每帧都会变
texture<float, 1> gridData1D;//整个程序里都不变
texture<unsigned char, 2> features2D;//整个程序里都不变
texture<float, 2> posteriors2D;//每一帧图像可能会变，因为有更新
texture<float, 1> tf1D;//每一帧图像会变，这是通过了方差过滤剩下的grid
texture<int,2> sumData2D;
texture<float,2>squmData2D;
texture<float,2>pEx2D;
texture<float,2>testData2D;
texture<float,2>nEx2D;
texture<float,2>example2D;
texture<float,2>batch2D;
//__constant__ char dev_features[6240];
//__constant__ float dev_lastb[4];
bool firstgetvar=true;
//bool first=true;
//bool firstlap=true;
////////////////////
//cudaArray* featuresArray;
//cudaArray* imageArray;
//cudaArray * posteriorsArray;
cudaArray* sumArray;
cudaArray* squmArray;
float *dev_grid;
float *dev_posteriors;
float *dev_pCounter;
float *dev_nCounter;
float *image;//尝试更改为cudaMallocPich和cudaMemcpy2D
float *dev_features;
float *dev_pEx;
float *dev_testData;
float *dev_nEx;
float *dev_example;
float *dev_batch;
float *dev_batchMatch;
float *dev_smatch;//存放模版匹配的额结果
float *dev_upPos;
int   *dev_upPosInd;
size_t img_pitch;
size_t pos_pitch;
size_t fea_pitch;
size_t pEx_pitch;
size_t testData_pitch;
size_t nEx_pitch;
size_t example_pitch;
size_t batch_pitch;
////////////////////
float* gridans;//getlappingker核函数返回的结果
float *tfans;//varfilter核函数返回的数据，每一帧的长度都是64034，所以不释放，第二次调用时也不用再分配内存。
float *withVarans;//gpu返回的结果，12行64034列
float *swithVarans;//返回到cpu的结果
float *dev_varisPass;//方差过滤的值，有两个核函数要用，所以设为全局变量
float *dev_filter1Ans;//一个核函数的结果，另一个核函数要用
float filter1_threshold;
///////////////////
//int img_data_size;
int img_w,img_h,gridl,gridw,maxThreads;
int sfeaturesw, sfeaturesh;
int postw,posth,patch_size;
int pEx_size=0,nEx_size=0;//记住正负模版数量
int ThreadNUM;
int dev_structSize;
int dev_nstructs;
__constant__ int patch[2];//patchSize常量内存，核函数经常调用，并且把平方也传入

__constant__ float conthreshold;
__constant__ int congridw;
__constant__ int connstructs;
__constant__ int constructSize;

//核函数：对一个grid做运算
__global__  void filter1ker(float *ans,int h,float threshold,int grid_w,int nstructs,int structSize)
{	
	//线程id
	//	int w=nstructs+2;
	int	tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid>=h) return;
	//不这样读，直接在大图上。用到这个图，其实也就在特征比对上用到了。
	int index=tex1Dfetch(tf1D, tid);//得到当前线程要处理的grid里的第几个box.
	/*
	int box_x=tex1Dfetch(gridData1D,index*5);
	int box_y=tex1Dfetch(gridData1D,index*5+1);
	int box_w=tex1Dfetch(gridData1D,index*5+2);
	int box_h=tex1Dfetch(gridData1D,index*5+3);
	int scale_idx=tex1Dfetch(gridData1D,index*5+4);
	*/
	//当不是四个四个存时读取方式也要变

	int box_x=tex1Dfetch(gridData1D,index);
	int box_y=tex1Dfetch(gridData1D,index+grid_w);
	int box_w=tex1Dfetch(gridData1D,index+grid_w*2);
	int box_h=tex1Dfetch(gridData1D,index+grid_w*3);
	int scale_idx=tex1Dfetch(gridData1D,index+grid_w*4);
	////////////	
	//计算ferns，cpu程序里这句classifier.getFeatures(patch,grid[i].sidx,ferns);
	int leaf;
	int x1,x2,y1,y2;
	//	int nstructs=10;
	//	int structSize =13;
	int imbig=0;
	float votes = 0;
	float point1,point2;
	for (int t=0;t<nstructs;t++){
		leaf=0;
		for (int f=0; f<structSize; f++){
			//取得点对坐标
			//	x1=tex2D(features2D,t*structSize*4+f*4,scale_idx);
			//	y1=tex2D(features2D,t*structSize*4+f*4+1,scale_idx);
			//	x2=tex2D(features2D,t*structSize*4+f*4+2,scale_idx);
			//	y2=tex2D(features2D,t*structSize*4+f*4+3,scale_idx);
			x1=tex2D(features2D,t*structSize+f,scale_idx);
			y1=tex2D(features2D,t*structSize+structSize*nstructs+f,scale_idx);
			x2=tex2D(features2D,t*structSize+structSize*nstructs*2+f,scale_idx);
			y2=tex2D(features2D,t*structSize+structSize*nstructs*3+f,scale_idx);

			point1=tex2D(imageData2D,box_x+x1,box_y+y1);//第y1行，第x1列，cpu版即这个意思important
			point2=tex2D(imageData2D,box_x+x2,box_y+y2);
			//  if(patch[x1*box_w+y1]>patch[x2*box_w+y2])
			if(point1>point2)
				imbig=1;
			else
				imbig=0;
			leaf = (leaf<<1 )+ imbig;
		}
		// ferns[t]=leaf;
		ans[tid*(nstructs+2)+t]=leaf;
		votes += tex2D(posteriors2D,leaf,t);

	}
	float conf=votes;
	ans[tid*(nstructs+2)+nstructs]=conf;
	if(conf>threshold)
		ans[tid*(nstructs+2)+nstructs+1]=index;
	else
		ans[tid*(nstructs+2)+nstructs+1]=-1;

}
//*/
__global__ void varfilterker(float *tfans,int grid_w,float var){
	//每个核函数计算一个box
	int	tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid>=grid_w) return;
	int index=tid;
	int box_x=tex1Dfetch(gridData1D,index);
	int box_y=tex1Dfetch(gridData1D,index+grid_w);
	int box_w=tex1Dfetch(gridData1D,index+grid_w*2);
	int box_h=tex1Dfetch(gridData1D,index+grid_w*3);
	int scale_idx=tex1Dfetch(gridData1D,index+grid_w*4);
	float brs =tex2D(sumData2D,box_x+box_w,box_y+box_h);//double brs = sum.at<int>(box.y+box.height,box.x+box.width);
	//sum.at<int>(x,y)取的是第x行第y列，tex2D(sumData2D,x,y)取的是第y行第x列
	float bls =tex2D(sumData2D,box_x,box_y+box_h);	//double bls = sum.at<int>(box.y+box.height,box.x);
	float trs =tex2D(sumData2D,box_x+box_w,box_y);//double trs = sum.at<int>(box.y,box.x+box.width);
	float tls =tex2D(sumData2D,box_x,box_y);//double tls = sum.at<int>(box.y,box.x);
	float brsq =tex2D(squmData2D,box_x+box_w,box_y+box_h);//double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
	float blsq =tex2D(squmData2D,box_x,box_y+box_h);	//double blsq = sqsum.at<double>(box.y+box.height,box.x);
	float trsq =tex2D(squmData2D,box_x+box_w,box_y);//double trsq = sqsum.at<double>(box.y,box.x+box.width);
	float tlsq =tex2D(squmData2D,box_x,box_y);//double tlsq = sqsum.at<double>(box.y,box.x);
	float mean = (brs+tls-trs-bls)/((float)box_w*box_h);//double mean = (brs+tls-trs-bls)/((double)box.area());
	float sqmean = (brsq+tlsq-trsq-blsq)/((float)box_w*box_h);//double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
	float temp=sqmean-mean*mean;	//return sqmean-mean*mean;
	if(temp>=var)
		tfans[tid]=tid;
	else
		tfans[tid]=-1;
}
float * Allvarfiltercu(const int *ssum,float *ssqum,int w,int h,float var){
	//绑定到纹理
	int ssum_data_size = sizeof(float) * w*h;
	int ssqum_data_size = sizeof(float) * w*h;
	if(firstgetvar)
	{
		cudaChannelFormatDesc chDesc6 = cudaCreateChannelDesc<int>();	
		cudaChannelFormatDesc chDesc7 = cudaCreateChannelDesc<float>();	
		cudaMallocArray(&sumArray, &chDesc6, w, h);
		cudaMallocArray(&squmArray, &chDesc7, w, h);
	}
	cudaMemcpyToArray( sumArray, 0, 0, ssum, ssum_data_size, cudaMemcpyHostToDevice);	
	cudaMemcpyToArray( squmArray, 0, 0, ssqum, ssqum_data_size, cudaMemcpyHostToDevice);
	if(firstgetvar){
		cudaBindTextureToArray( sumData2D, sumArray);	
		cudaBindTextureToArray( squmData2D, squmArray);
	}
	//让核函数去做，每个线程做一个box
	dim3 blocks((gridw+255)/256);
	dim3 threads(256);
	varfilterker<<<blocks,threads>>>(tfans,gridw,var);
	cudaThreadSynchronize();	
	float *stfans=new float[gridw];//待会谁用谁释放。
	cudaMemcpy( stfans, tfans, gridw*sizeof(float), cudaMemcpyDeviceToHost);
	firstgetvar=false;
	return stfans;
}
void filter1cucha(const unsigned char *simg, float *varisPass,float * filter1Ans,int varis_index)
{
	cudaMemcpy2D(image, img_pitch, simg, sizeof(unsigned char) * img_w, sizeof(unsigned char) * img_w, img_h, cudaMemcpyHostToDevice);
	//把varisPass复制过去 
	cudaMemcpy( dev_varisPass, varisPass, varis_index * sizeof( float ), cudaMemcpyHostToDevice );	
	///////////////////////
	//cudaMemcpy2D(dev_posteriors, pos_pitch, sposteriors, sizeof(float) * postw, sizeof(float) * postw, posth, cudaMemcpyHostToDevice);
	//float *ans;//filter1ker核函数返回的数据，每一帧图像都要释放，因为每一帧时varis_index值不一样，所以我们需要的ans的大小不一样
	//12*varis_index大小，前10存ferns,第11个存measure_forest的值conf，第12个,如果conf大于阈值，则存索引i否则存-1,varis_index是tf的长度
	//接下来计划每个线程处理一个网格	,核函数要输出ferns,conf,i
	dim3 blocks((varis_index+255)/256);
	dim3 threads(256);		
	//申请的数组的大小是(nstructs+2)*varis_index大小
	filter1ker<<< blocks, threads>>>( dev_filter1Ans, varis_index,filter1_threshold,gridw,dev_nstructs,dev_structSize);
	cudaThreadSynchronize();
	cudaMemcpy(filter1Ans, dev_filter1Ans, (dev_nstructs+2)*varis_index*sizeof(float) , cudaMemcpyDeviceToHost);
	//cudaMemcpy(out,dev_posteriors,8192*10*4,cudaMemcpyDeviceToHost);测试时用来传出显存数据查看用的
}

__global__  void filter1ker1(float *ans,int varis_index)
{	
	extern __shared__ float shared[];
	//	int w=nstructs+2;
	int tidx=threadIdx.x;
	int tidy=threadIdx.y;//待会要从features2D上取第tidx组，tidy个
	int tid=threadIdx.x*blockDim.y+threadIdx.y;//int nstructs,int structSize这两个变量不用再传了，可以从这取
	//int bid=blockIdx.x;
	int bid=blockIdx.x*gridDim.y+blockIdx.y;
	if(bid>=varis_index) return;
	int index=tex1Dfetch(tf1D, bid);//得到当前线程要处理的grid里的第几个box.
	//当不是四个四个存时读取方式也要变
	int box_x=tex1Dfetch(gridData1D,index);
	int box_y=tex1Dfetch(gridData1D,index+congridw);
	int box_w=tex1Dfetch(gridData1D,index+congridw*2);
	int box_h=tex1Dfetch(gridData1D,index+congridw*3);
	int scale_idx=tex1Dfetch(gridData1D,index+congridw*4);
	int i;
	////////////	
	//计算ferns
	shared[tid]=0;
	int x1,x2,y1,y2;
	float point1,point2;	
	x1=tex2D(features2D,tidx*constructSize+tidy,scale_idx);
	y1=tex2D(features2D,tidx*constructSize+constructSize*connstructs+tidy,scale_idx);
	x2=tex2D(features2D,tidx*constructSize+constructSize*connstructs*2+tidy,scale_idx);
	y2=tex2D(features2D,tidx*constructSize+constructSize*connstructs*3+tidy,scale_idx);
	point1=tex2D(imageData2D,box_x+x1,box_y+y1);//第y1行，第x1列，cpu版即这个意思important
	point2=tex2D(imageData2D,box_x+x2,box_y+y2);
	if(point1>point2)
		shared[tid]=1<<(constructSize-1-tidy);//为算leaf准备

	__syncthreads();
	////////////////////////////////////////////优化////////////////////////////////////////////////////////
	///*
	for(i=blockDim.y;i>1;){
		if(tidy<i/2){
			i=(i+1)/2;
			shared[tid]+=shared[tid+i];
		}
		else
			break;
		__syncthreads();
	}
	if(tidy==0)
	{
		ans[bid*(connstructs+2)+tidx]=shared[tid];//把leaf写入。**********************************
	}
	if(tidy==1)
	{
		shared[tid]= tex2D(posteriors2D,shared[tid-1],tidx);//算每个votes（总共10个），每行的第二个线程计算，算完以后存到第二线程的shared，不存到第一个线程的shared吗，否则还要同步
	}
	__syncthreads();
	//leaf是一行线程算，conf是一个block算，比较出结果也是一个block算，都是tid=0的算，千万别让多个线程重合的算

	if(tid==1)
	{
		for(i=1;i<connstructs;i++)
			shared[1]+=shared[i*constructSize+1];

	}	
	/*下面这段优化结果不对，不知道什么问题
	if(tidy==1)//刚才把votes都存储到每行的第二列shared了
	{
	for(i=blockDim.x;i>1;)
	{
	if(tidx<i/2)
	{
	i=(i+1)/2;
	shared[tid]+=shared[tid+i*blockDim.y];
	}
	else
	break;
	__syncthreads();
	}
	}*/
	__syncthreads();
	if(tid==0)
		ans[bid*(connstructs+2)+connstructs]=shared[1];//
	if(tid==1)
	{
		if(shared[1]>conthreshold)
			ans[bid*(connstructs+2)+connstructs+1]=index;//*********************************
		else
			ans[bid*(connstructs+2)+connstructs+1]=-1;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////
	//下面是原来的，上面是优化的
	/*
	if(tidy==0)
	{
	for(i=1;i<constructSize;i++)
	shared[tid]+=shared[tid+i];//算leaf，把线程块（10行（组）13列），每行13个（每个线程算一个）都加到第一个线程	
	ans[bid*(connstructs+2)+tidx]=shared[tid];//把leaf写入。**********************************
	shared[tid]= tex2D(posteriors2D,shared[tid],tidx);//算每个votes（总共10个）
	}
	__syncthreads();
	//leaf是一行线程算，conf是一个block算，比较出结果也是一个block算，都是tid=0的算，千万别让多个线程重合的算
	if(tid==0)
	{
	for(i=1;i<connstructs;i++)
	shared[0]+=shared[i*constructSize];
	ans[bid*(connstructs+2)+connstructs]=shared[0];//*********************************conf
	if(shared[0]>conthreshold)
	ans[bid*(connstructs+2)+connstructs+1]=index;//*********************************
	else
	ans[bid*(connstructs+2)+connstructs+1]=-1;
	}
	*/
	///////////////////////////////////////////////////////////////////////////////
}

void filter1cu(unsigned char *simg, float *varisPass,float * filter1Ans,int varis_index,double &f1datacost)
{   //优化filter1cu，所以程序的输入输出还是一样的，只不过线程的组织结构变了
	//cudaHostRegister((void*)simg,sizeof(unsigned char) * img_w*img_h,1);
	cudaMemcpy2D(image, img_pitch, simg, sizeof(unsigned char) * img_w, sizeof(unsigned char) * img_w, img_h, cudaMemcpyHostToDevice);
	//把varisPass复制过去 
	cudaMemcpy( dev_varisPass, varisPass, varis_index * sizeof( float ), cudaMemcpyHostToDevice );	
	///////////////////////
	//cudaMemcpy2D(dev_posteriors, pos_pitch, sposteriors, sizeof(float) * postw, sizeof(float) * postw, posth, cudaMemcpyHostToDevice);
	//float *ans;//filter1ker核函数返回的数据，每一帧图像都要释放，因为每一帧时varis_index值不一样，所以我们需要的ans的大小不一样
	//(nstructs+2)*varis_index大小，前10存ferns,第11个存measure_forest的值conf，第12个,如果conf大于阈值，则存索引i否则存-1,varis_index是tf的长度
	//接下来计划每个线程处理一个网格	,核函数要输出ferns,conf,i
	//dim3 blocks(varis_index);//每个块算一个图
	double fangx= sqrt((double)varis_index);
	int fang= int(fangx);
	//dim3 blocks(varis_index/65535+1,65535);//每个块算一个图
	dim3 blocks(varis_index/fang+1,fang);
	dim3 threads(dev_nstructs,dev_structSize);//每个线程算一个对点，每一行线程算一组，每组由13个点对组成。//分为dev_nstructs行的线程，dev_structSize列的线程		
	//申请的数组的大小是(nstructs+2)*varis_index大小
	filter1ker1<<< blocks, threads,dev_nstructs*dev_structSize*sizeof(float)>>>( dev_filter1Ans,varis_index);
	cudaThreadSynchronize();
	cudaMemcpy(filter1Ans, dev_filter1Ans, (dev_nstructs+2)*varis_index*sizeof(float) , cudaMemcpyDeviceToHost);
	//cudaMemcpy(out,dev_posteriors,8192*10*4,cudaMemcpyDeviceToHost);测试时用来传出显存数据查看用的
	f1datacost=0;
}

__global__ void fiterWithVarker(float *ans,int grid_w,float var,int w,float threshold)
{
	int	tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid>=grid_w) return;
	int index=tid;
	int box_x=tex1Dfetch(gridData1D,index);
	int box_y=tex1Dfetch(gridData1D,index+grid_w);
	int box_w=tex1Dfetch(gridData1D,index+grid_w*2);
	int box_h=tex1Dfetch(gridData1D,index+grid_w*3);
	int scale_idx=tex1Dfetch(gridData1D,index+grid_w*4);
	float brs =tex2D(sumData2D,box_x+box_w,box_y+box_h);//double brs = sum.at<int>(box.y+box.height,box.x+box.width);
	//sum.at<int>(x,y)取的是第x行第y列，tex2D(sumData2D,x,y)取的是第y行第x列
	float bls =tex2D(sumData2D,box_x,box_y+box_h);	//double bls = sum.at<int>(box.y+box.height,box.x);
	float trs =tex2D(sumData2D,box_x+box_w,box_y);//double trs = sum.at<int>(box.y,box.x+box.width);
	float tls =tex2D(sumData2D,box_x,box_y);//double tls = sum.at<int>(box.y,box.x);
	float brsq =tex2D(squmData2D,box_x+box_w,box_y+box_h);//double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
	float blsq =tex2D(squmData2D,box_x,box_y+box_h);	//double blsq = sqsum.at<double>(box.y+box.height,box.x);
	float trsq =tex2D(squmData2D,box_x+box_w,box_y);//double trsq = sqsum.at<double>(box.y,box.x+box.width);
	float tlsq =tex2D(squmData2D,box_x,box_y);//double tlsq = sqsum.at<double>(box.y,box.x);
	float mean = (brs+tls-trs-bls)/((float)box_w*box_h);//double mean = (brs+tls-trs-bls)/((double)box.area());
	float sqmean = (brsq+tlsq-trsq-blsq)/((float)box_w*box_h);//double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
	//float temp=sqmean-mean*mean;	//return sqmean-mean*mean;
	if((sqmean-mean*mean)<var)
	{
		ans[tid*w+11]=-2;
		return;
	}
	////////////	
	//计算ferns，cpu程序里这句classifier.getFeatures(patch,grid[i].sidx,ferns);
	int leaf;
	int x1,x2,y1,y2;
	int nstructs=10;
	int structSize =13;
	int imbig=0;
	float votes = 0;
	float point1,point2;
	for (int t=0;t<nstructs;t++){
		leaf=0;
		for (int f=0; f<structSize; f++){
			//取得点对坐标
			x1=tex2D(features2D,t*structSize*4+f*4,scale_idx);
			y1=tex2D(features2D,t*structSize*4+f*4+1,scale_idx);
			x2=tex2D(features2D,t*structSize*4+f*4+2,scale_idx);
			y2=tex2D(features2D,t*structSize*4+f*4+3,scale_idx);
			point1=tex2D(imageData2D,box_x+x1,box_y+y1);//第y1行，第x1列，cpu版即这个意思important
			point2=tex2D(imageData2D,box_x+x2,box_y+y2);
			//  if(patch[x1*box_w+y1]>patch[x2*box_w+y2])
			if(point1>point2)
				imbig=1;
			else
				imbig=0;
			leaf = (leaf<<1 )+ imbig;
		}
		// ferns[t]=leaf;
		ans[tid*w+t]=leaf;
		votes += tex2D(posteriors2D,leaf,t);
	}	
	ans[tid*w+10]=votes;
	if(votes>threshold)
		ans[tid*w+11]=index;
	else
		ans[tid*w+11]=-1;
	return;		
}
float *fiterWithVarcu(const int *ssum,float *ssqum,int w,int h,float var,const unsigned char *simg, float threshold, float *sposteriors)
{
	int ssum_data_size = sizeof(float) * w*h;
	int ssqum_data_size = sizeof(float) * w*h;
	if(firstgetvar)
	{
		cudaChannelFormatDesc chDesc6 = cudaCreateChannelDesc<int>();	
		cudaChannelFormatDesc chDesc7 = cudaCreateChannelDesc<float>();	
		cudaMallocArray(&sumArray, &chDesc6, w, h);
		cudaMallocArray(&squmArray, &chDesc7, w, h);
	}
	cudaMemcpyToArray( sumArray, 0, 0, ssum, ssum_data_size, cudaMemcpyHostToDevice);	
	cudaMemcpyToArray( squmArray, 0, 0, ssqum, ssqum_data_size, cudaMemcpyHostToDevice);
	if(firstgetvar){
		cudaBindTextureToArray( sumData2D, sumArray);	
		cudaBindTextureToArray( squmData2D, squmArray);
	}
	//让核函数去做，每个线程做一个box
	cudaMemcpy2D(image, img_pitch, simg, sizeof(unsigned char) * img_w, sizeof(unsigned char) * img_w, img_h, cudaMemcpyHostToDevice);
	cudaMemcpy2D(dev_posteriors, pos_pitch, sposteriors, sizeof(float) * postw, sizeof(float) * postw, posth, cudaMemcpyHostToDevice);

	dim3 blocks((gridw+255)/256);
	dim3 threads(256);
	fiterWithVarker<<<blocks,threads>>>(withVarans,gridw,var,12,threshold);		
	cudaThreadSynchronize();
	cudaMemcpy(swithVarans, withVarans, 12*gridw*sizeof(float) , cudaMemcpyDeviceToHost);//待会调用者释放
	firstgetvar=false;
	return swithVarans;
}
__global__ void getlappingker(float box1_x,float box1_y,float box1_w,float box1_h, int grid_w,float *gridans,float * dev_grid)
{	
	int	tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid>=grid_w)
		return;
	float box2_x=tex1Dfetch(gridData1D,tid);
	float box2_y=tex1Dfetch(gridData1D,tid+grid_w);
	float box2_w=tex1Dfetch(gridData1D,tid+grid_w*2);
	float box2_h=tex1Dfetch(gridData1D,tid+grid_w*3);
	if((box1_x > box2_x+box2_w)||(box1_y > box2_y+box2_h)||(box1_x+box1_w < box2_x)||(box1_y+box1_h < box2_y)){
		dev_grid[tid+grid_w*5]=0;		
		gridans[tid]=0;
		return;
	}
	float colInt =  min(box1_x+box1_w,box2_x+box2_w) - max(box1_x, box2_x);
	float rowInt =  min(box1_y+box1_h,box2_y+box2_h) - max(box1_y,box2_y);
	float intersection = colInt * rowInt;
	float area1 = box1_w*box1_h;
	float area2 = box2_w*box2_h;
	float answ=intersection / (area1 + area2 - intersection);
	dev_grid[tid+grid_w*5]=answ;
	gridans[tid]=answ;
	return;
}

//输出一个10大小的goodbox,和bad_box
void getlappingcu(float * lastb,float *sgridans)
{
	dim3 blocks((gridw+255)/256);
	dim3 threads(256);
	getlappingker<<<blocks,threads>>>(lastb[0],lastb[1],lastb[2],lastb[3],gridw,gridans,dev_grid);
	cudaMemcpy(sgridans, gridans, sizeof(float)*gridw, cudaMemcpyDeviceToHost);//把核函数的计算结果复制回cpu
}
//专门加入一个init方法处理那些一次性处理的东西,分配显存
//下面要分两步来做learn，首先把getOverlapping,同时做了bad_box的更新，待会再用个核函数做了good_box的更新
void gpuParam(int nstructs ,int structSize){
	//如果structSize超过23时，pow(2.0,structSize)这个浮点数可能会超过init所能表示的范围，所以cpu版的程序一旦structSize一旦超过23，就会报错了
	postw=pow(2.0,structSize);
	dev_nstructs=nstructs;
	dev_structSize=structSize;
	posth=nstructs;
	cudaMemcpyToSymbol((char *) &connstructs,(void *)&dev_nstructs,sizeof(int));//加载到常量内存里
	cudaMemcpyToSymbol((char *) &constructSize,(void *)&dev_structSize,sizeof(int));//加载到常量内存里
}
void initNccData(int patch_s) //必须单独放一个Ncc的数据初始化，比如分配纹理，分配显存，因为在tld.init()里就开始更新pEx和nEx并且使用NCConf了
{  //这个函数在tld的构造函数和tld.init之间运行
	int patchs[2];
	patchs[0]=patch_s;
	patchs[1]=patch_s*patch_s;
	patch_size=patch_s;
	cudaMemcpyToSymbol((char *) patch,(void *)patchs,sizeof(int)*2);//加载到常量内存里
	//绑定两个纹理，大小都是patch_size*patch_size*100，pEx，nEx
	cudaMallocPitch((void**)(&dev_pEx), &pEx_pitch, sizeof(float) * patch_size, patch_size*100);//程序里是float，所以这里也只处理float类型的。
	cudaChannelFormatDesc pExDesc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, pEx2D, dev_pEx, pExDesc, patch_size, patch_size*100, pEx_pitch);
	//nEx
	cudaMallocPitch((void**)(&dev_nEx), &nEx_pitch, sizeof(float) * patch_size, patch_size*100);
	cudaChannelFormatDesc nExDesc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, nEx2D, dev_nEx, nExDesc, patch_size, patch_size*100, nEx_pitch);
	//
	//batch，批量的example,最多100个
	cudaMallocPitch((void**)(&dev_batch), &batch_pitch, sizeof(float) * patch_size, patch_size*100);
	cudaChannelFormatDesc batchDesc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, batch2D, dev_batch, batchDesc, patch_size, patch_size*100, batch_pitch);
	//posteriors的初始化，而且cpu每次更新都有函数更新到gpu，所以不用再传了
	cudaMallocPitch((void**)(&dev_posteriors), &pos_pitch, sizeof(float) * postw, posth);
	cudaChannelFormatDesc posDesc = cudaCreateChannelDesc<float>();
	cudaMemset(dev_posteriors,0,pos_pitch*posth);//初始化值,如果是随着每个posteriors的更新而更新，就要这个初始化0
	//cudaMemset2D(dev_posteriors,pos_pitch,0,sizeof(float) * postw,posth);//初始化值,如果是随着每个posteriors的更新而更新，就要这个初始化0	
	cudaBindTexture2D(NULL, posteriors2D, dev_posteriors, posDesc, postw, posth, pos_pitch);
	//example,即我们要处理的模版
	cudaMallocPitch((void**)(&dev_example), &example_pitch, sizeof(float) * patch_size, patch_size);
	cudaChannelFormatDesc exampleDesc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, example2D, dev_example, exampleDesc, patch_size, patch_size, example_pitch);
	cudaMalloc((void **) &dev_smatch,200*sizeof(float));//给模板匹配gpu的核函数分配一个存储结果的显存，一次分配大一点显存，这样不用每次都分配显存，节省时间
	cudaMalloc((void **) &dev_batchMatch,200*100*sizeof(float));//给批处理图像的ker分配一个存储结果的全局显存，至少要(pEx.size()+nEx.size())*detections大小，一次分配，重复利用，不用每次都分配，节省时间
	//分配update，posterios数组用的大小为nstructs*2的float数组
	cudaMalloc((void **) &dev_upPos,dev_nstructs*sizeof(float));
	cudaMalloc((void **) &dev_upPosInd,dev_nstructs*sizeof(float));
}
void initGpuData(float filter1threshold,int img_ww,int img_hh,float *sgrid ,int gridww,unsigned char *sfeatures,int sfeaturesww, int sfeatureshh)
{	
	filter1_threshold=filter1threshold;
	img_w=img_ww;
	img_h=img_hh;
	gridw=gridww;
	gridl=gridww*6;
	sfeaturesw=sfeaturesww;
	sfeaturesh=sfeatureshh;
	cudaMemcpyToSymbol((char *) &conthreshold,(void *)&filter1_threshold,sizeof(float));//加载到常量内存里
	cudaMemcpyToSymbol((char *) &congridw,(void *)&gridw,sizeof(int));//加载到常量内存里

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);	
	maxThreads=prop.maxThreadsPerBlock;


	//给img绑定纹理，过滤器1里要用
	cudaMallocPitch((void**)(&image), &img_pitch, sizeof(unsigned char) * img_w, img_h);
	cudaChannelFormatDesc imgDesc = cudaCreateChannelDesc<unsigned char>();
	cudaBindTexture2D(NULL, imageData2D, image, imgDesc, img_w, img_h, img_pitch);
	///////////////////////////////////////
	//grid
	int grid_data_size = sizeof(float) * gridl;
	cudaMalloc((void**)&dev_grid,grid_data_size);
	cudaMemcpy(dev_grid,sgrid,grid_data_size,cudaMemcpyHostToDevice);
	cudaBindTexture(0,gridData1D,dev_grid);
	////////////////////////////////////
	//feature
	cudaMallocPitch((void**)(&dev_features), &fea_pitch, sizeof(unsigned char) * sfeaturesw, sfeaturesh);
	cudaChannelFormatDesc feaDesc = cudaCreateChannelDesc<unsigned char>();
	cudaMemcpy2D(dev_features, fea_pitch, sfeatures, sizeof(unsigned char) * sfeaturesw, sizeof(unsigned char) * sfeaturesw, sfeaturesh, cudaMemcpyHostToDevice);
	cudaBindTexture2D(NULL, features2D, dev_features, feaDesc, sfeaturesw, sfeaturesh, fea_pitch);
	/////////////////////////////////////
	/*
	//posteriors
	cudaMallocPitch((void**)(&dev_posteriors), &pos_pitch, sizeof(float) * postw, posth);
	cudaChannelFormatDesc posDesc = cudaCreateChannelDesc<float>();
	for(int i=0;i<posth;i++)
	cudaMemcpy2D((char*)dev_posteriors+i*pos_pitch, pos_pitch, posteriorsP[i], sizeof(float) * postw, sizeof(float) * postw, 1, cudaMemcpyHostToDevice);
	cudaBindTexture2D(NULL, posteriors2D, dev_posteriors, posDesc, postw, posth, pos_pitch);
	//////////////////////////////////
	*/
	cudaMalloc( (void**)&dev_varisPass, gridw * sizeof( float ) );//存放方差过滤后的数据，这个数据从cpu传来，待会要存储为纹理，供ker使用，最大不超过gridw，所以一次分配，节省时间
	cudaBindTexture(0, tf1D, dev_varisPass);
	/*
	cudaMallocPitch((void**)(&dev_pCounter), &pCo_pitch, sizeof(float) * postw, posth);
	cudaChannelFormatDesc pCoDesc = cudaCreateChannelDesc<float>();
	cudaMemcpy2D(dev_pCounter, pCo_pitch, spCounter, sizeof(float) * postw, sizeof(float) * postw, posth, cudaMemcpyHostToDevice);	
	cudaBindTexture2D(NULL, pCounter2D, dev_pCounter, pCoDesc, postw, posth, pCo_pitch);

	/////////////////////////////////
	cudaMallocPitch((void**)(&dev_nCounter), &nCo_pitch, sizeof(float) * postw, posth);
	cudaChannelFormatDesc nCoDesc = cudaCreateChannelDesc<float>();
	cudaMemcpy2D(dev_nCounter, nCo_pitch, snCounter, sizeof(float) * postw, sizeof(float) * postw, posth, cudaMemcpyHostToDevice);	
	cudaBindTexture2D(NULL, nCounter2D, dev_nCounter, nCoDesc, postw, posth, nCo_pitch);
	/////////////////////////////////
	*/
	//gridans
	cudaMalloc( (void**)&gridans, sizeof(float)*gridw);	//给核函数的结果分配空间存储
	cudaMalloc( (void**)&tfans, sizeof(float)*gridw );	//给varfilterker核函数的结果分配空间存储    
	cudaMalloc( (void**)&withVarans, 12*gridw*sizeof(float) );
	swithVarans=new float[12*gridw*sizeof(float) ];//每次图像不一样分配的内存也不一样，所以不能设为全局

	/////////////////////////////////
	cudaMalloc( (void**)&dev_filter1Ans, (dev_nstructs+2)*gridw*sizeof(float) );//由于varis_index的大小是变的，但我们可以申请大点的内存。

}
//释放显存
void endGpuData()
{
	cudaUnbindTexture( imageData2D );
	cudaUnbindTexture( gridData1D );
	cudaUnbindTexture(features2D );
	cudaUnbindTexture( posteriors2D );	
	cudaUnbindTexture( tf1D );
	cudaUnbindTexture(pEx2D);
	cudaUnbindTexture(nEx2D);
	cudaUnbindTexture(example2D);
	cudaUnbindTexture(batch2D);
	cudaFree( dev_features );
	cudaFree( dev_posteriors);
	cudaFree(dev_pEx);
	cudaFree(dev_nEx);
	cudaFree(dev_example);
	cudaFree(dev_batch);
	cudaFree(image);
	cudaFree(dev_grid);
	cudaFree(gridans);
	cudaFree(tfans);
	cudaFree(withVarans);
	cudaFree(dev_smatch);
	cudaFree(dev_batchMatch);	
	cudaFree(dev_varisPass);
	cudaFree(dev_filter1Ans);
	cudaFree(dev_upPos);
	cudaFree(dev_upPosInd);
	pEx_size=0,nEx_size=0;
	delete [] swithVarans;
}

__global__ void updatePoker(float *dev_posteriors,int *dev_upPosInd,float* dev_upPos,int pos_pitch )
{
	//int threadNum=blockDim.x;
	int tid=threadIdx.x;
	int idx=dev_upPosInd[tid];
	float var=dev_upPos[tid];
	*((float*)((char *)dev_posteriors+pos_pitch*tid)+idx)=var;
}

void updatePoscu(float *upPos,int *upPosInd)
{
	cudaMemcpy(dev_upPos,upPos,dev_nstructs*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_upPosInd,upPosInd,dev_nstructs*sizeof(int),cudaMemcpyHostToDevice);
	updatePoker<<<1,dev_nstructs,0>>>(dev_posteriors,dev_upPosInd,dev_upPos,pos_pitch);
	//	*(float*)((char *)dev_posteriors+pos_pitch*2+143)=0.9;
}
void addpExcu(const float *spEx)//把更新的pEx数据加载到gpu
{
	cudaMemcpy2D((float*)((char*)dev_pEx+patch_size*pEx_size*pEx_pitch), pEx_pitch, spEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	pEx_size++;
}

void addpExcu(const float *spEx,int position,bool full)
{
	if(!full)
	{
		cudaMemcpy2D((float*)((char*)dev_pEx+patch_size*pEx_size*pEx_pitch), pEx_pitch, spEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	pEx_size++;	
	}
	else
	{
		cudaMemcpy2D((float*)((char*)dev_pEx+patch_size*position*pEx_pitch), pEx_pitch, spEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	

	}
}


void addnExcu1(const float *snEx)//把更新的nEx数据加载到gpu
{	
	cudaMemcpy2D((float*)((char*)dev_nEx+patch_size*nEx_size*nEx_pitch), nEx_pitch, snEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	nEx_size++;	

}
void addnExcu(const float *snEx)//把更新的nEx数据加载到gpu
{	
	cudaMemcpy2D((float*)((char*)dev_nEx+patch_size*nEx_size*nEx_pitch), nEx_pitch, snEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	nEx_size++;	

}
void addnExcu(const float *snEx,int position,bool full)
{
	if(!full)
	{
		cudaMemcpy2D((float*)((char*)dev_nEx+patch_size*nEx_size*nEx_pitch), nEx_pitch, snEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	nEx_size++;	
	}
	else
	{
		cudaMemcpy2D((float*)((char*)dev_nEx+patch_size*position*nEx_pitch), nEx_pitch, snEx, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	

	}
}
__global__  void NNConfker(float *dev_smatch,int pEx_s)//注意核函数只能访问显存里的东西，其他参数要么通过参数传递或者全局内存
{   

	extern __shared__ float shared[];
	int tid=threadIdx.x;//blockDim.x=patch_size
	int bid=blockIdx.x;
	int threadNum=blockDim.x;
	int photox;;//第photox行
	int photoy;;//第photo列
	int  offset=(threadNum+1)/2;;
	//int offset =(patch_size+1)/2;
	float pi,ei;
	int i;
	shared[tid*3]=0;
	shared[tid*3+1]=0;
	shared[tid*3+2]=0;
	if(bid<pEx_s){//说明这个block该处理正模版库和example的对比
		for(i=tid;i<patch[1];i+=threadNum)//一个线程可能要处理好几个图像点
		{
			photox=i/patch[0];
			photoy=i%patch[0];
			ei=tex2D(example2D,photoy,photox);
			pi=tex2D(pEx2D,photoy,bid*patch[0]+photox);
			shared[tid*3]+=ei*pi;
			shared[tid*3+1]+=ei*ei;
			shared[tid*3+2]+=pi*pi;
		}
		__syncthreads();
		////////////////////////////////////////////////////////////////
		for(i=blockDim.x;i>1;)
		{
			if(tid<i/2)
			{
				i=(i+1)/2;
				shared[tid*3] += shared[(tid+i)*3];
				shared[tid*3+1] += shared[(tid+i)*3+1];
				shared[tid*3+2] += shared[(tid+i)*3+2];
			}
			else 
				break;
			__syncthreads();
		}
		__syncthreads();
		/*
		while(offset>0) {
		if(tid<offset) {
		if(tid+offset<threadNum)
		{
		shared[tid*3] += shared[(tid+offset)*3];
		shared[tid*3+1] += shared[(tid+offset)*3+1];
		shared[tid*3+2] += shared[(tid+offset)*3+2];
		}
		}
		threadNum=offset;
		if(offset==1)
		offset=0;
		else
		offset =(offset+1)/2;

		__syncthreads();
		}
		*/
		if(tid<2)
			shared[tid+1]=sqrt((float) shared[tid+1]);
		__syncthreads();
		if(tid == 0) {
			shared[1]=shared[1]*shared[2];
			dev_smatch[bid] = (shared[0]/shared[1]+1)/2;  
		}
	}
	else
	{

		if(bid-pEx_s<0) return;//可能nex大小为0
		for(i=tid;i<patch[1];i+=threadNum)//一个线程可能要处理好几个图像点
		{
			photox=i/patch[0];
			photoy=i%patch[0];
			ei=tex2D(example2D,photoy,photox);
			pi=tex2D(nEx2D,photoy,(bid-pEx_s)*patch[0]+photox);//这里别忘啊减去pEx
			shared[tid*3]+=ei*pi;
			shared[tid*3+1]+=ei*ei;
			shared[tid*3+2]+=pi*pi;
		}
		__syncthreads();   
		////////////////////////////////////////////
		for(i=blockDim.x;i>1;)
		{
			if(tid<i/2)
			{
				i=(i+1)/2;
				shared[tid*3] += shared[(tid+i)*3];
				shared[tid*3+1] += shared[(tid+i)*3+1];
				shared[tid*3+2] += shared[(tid+i)*3+2];
			}
			else
				break;
			__syncthreads();
		}
		__syncthreads();
		//可优化
		/*
		while(offset>0) {
		if(tid<offset) {
		if(tid+offset<threadNum)
		{
		shared[tid*3] += shared[(tid+offset)*3];
		shared[tid*3+1] += shared[(tid+offset)*3+1];
		shared[tid*3+2] += shared[(tid+offset)*3+2];
		}
		}
		threadNum=offset;
		if(offset==1)
		offset=0;
		else
		offset =(offset+1)/2;


		__syncthreads();
		}
		*/
		if(tid<2)
			shared[tid+1]=sqrt((float) shared[tid+1]);
		__syncthreads();
		if(tid == 0) {
			shared[1]=shared[1]*shared[2];
			dev_smatch[bid] = (shared[0]/shared[1]+1)/2;  
		}
	}
}
void NNConfcu(const float *sexample,float *smatch)
{
	cudaMemcpy2D(dev_example, example_pitch, sexample, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	//把pEx和nEx的大小设成显存的常量，这里还是不这样，我们把pEx_size当成参数传给核函数
	dim3 blocks(pEx_size+nEx_size);//不管怎样我们都是让一个线程块处理一个小图。
	//dim3 threads(patch_size,patch_size);//要从gpu传来,相乘不能大于512，否则会出错，也即patch_size不能大于24，可以定死这里让块大小固定，而让一个块处理多个像素点
	if((patch_size*patch_size)>maxThreads)
		ThreadNUM=maxThreads;
	else
		ThreadNUM=patch_size*patch_size;//这样绝对不会出现多余的线程了
	dim3 threads(ThreadNUM);
	NNConfker<<< blocks, threads,ThreadNUM*sizeof(float)*3>>>(dev_smatch,pEx_size);
	cudaThreadSynchronize();
	cudaMemcpy(smatch,dev_smatch,(pEx_size+nEx_size)*sizeof(float),cudaMemcpyDeviceToHost);
}
__global__ void NNConfBatchker(float *dev_batchMatch,int pEx_s)
{
	//和单个example的是一样的，不同的是，取example时取的是batch2D的第blockIDx.x个图像
	//由于原来blockIDx是一维的所以只有x坐标，现在把x和y坐标互换一下就行了
	//写的时候呢写到一个和blockIdx相关的dev_batchMatch中


	extern __shared__ float shared[];
	int tid=threadIdx.x;//blockDim.x=patch_size
	int bid=blockIdx.y;
	int bidx=blockIdx.x;
	int threadNum=blockDim.x;
	int photox;;//第photox行
	int photoy;;//第photo列
	int  offset=(threadNum+1)/2;;
	//int offset =(patch_size+1)/2;
	float pi,ei;
	int i;
	shared[tid*3]=0;
	shared[tid*3+1]=0;
	shared[tid*3+2]=0;
	if(bid<pEx_s){//说明这个block该处理正模版库和example的对比
		for(i=tid;i<patch[1];i+=threadNum)//一个线程可能要处理好几个图像点
		{
			photox=i/patch[0];
			photoy=i%patch[0];
			ei=tex2D(batch2D,photoy,photox+bidx*patch[0]);//取batch2D上bidx个图，行数加上bidx*patch[0]
			pi=tex2D(pEx2D,photoy,bid*patch[0]+photox);
			shared[tid*3]+=ei*pi;
			shared[tid*3+1]+=ei*ei;
			shared[tid*3+2]+=pi*pi;
		}
		__syncthreads();
		////////////////////////////////////////////////////////////////////
		///*
		for(i=blockDim.x;i>1;)
		{
			if(tid<i/2)
			{
				i=(i+1)/2;
				shared[tid*3] += shared[(tid+i)*3];
				shared[tid*3+1] += shared[(tid+i)*3+1];
				shared[tid*3+2] += shared[(tid+i)*3+2];
			}
			else
				break;
			__syncthreads();
		}
		__syncthreads();
		//*/
		/*
		while(offset>0) {
		if(tid<offset) {
		if(tid+offset<threadNum)
		{
		shared[tid*3] += shared[(tid+offset)*3];
		shared[tid*3+1] += shared[(tid+offset)*3+1];
		shared[tid*3+2] += shared[(tid+offset)*3+2];
		}
		}
		threadNum=offset;
		if(offset==1)
		offset=0;
		else
		offset =(offset+1)/2;


		__syncthreads();
		}
		*/
		if(tid<2)
			shared[tid+1]=sqrt((float) shared[tid+1]);
		__syncthreads();
		if(tid == 0) {
			shared[1]=shared[1]*shared[2];
			dev_batchMatch[bidx*gridDim.y+bid] = (shared[0]/shared[1]+1)/2;  //写数据        
		}
	}
	else
	{

		if(bid-pEx_s<0) return;//可能nex大小为0
		for(i=tid;i<patch[1];i+=threadNum)//一个线程可能要处理好几个图像点
		{
			photox=i/patch[0];
			photoy=i%patch[0];
			ei=tex2D(batch2D,photoy,photox+bidx*patch[0]);
			pi=tex2D(nEx2D,photoy,(bid-pEx_s)*patch[0]+photox);//这里别忘啊减去pEx
			shared[tid*3]+=ei*pi;
			shared[tid*3+1]+=ei*ei;
			shared[tid*3+2]+=pi*pi;
		}
		__syncthreads();   
		///////////////////////////////////////////////////////
		///*
		for(i=blockDim.x;i>1;)
		{
			if(tid<i/2)
			{
				i=(i+1)/2;
				shared[tid*3] += shared[(tid+i)*3];
				shared[tid*3+1] += shared[(tid+i)*3+1];
				shared[tid*3+2] += shared[(tid+i)*3+2];
			}
			else
				break;
			__syncthreads();
		}
		__syncthreads();
		//*/
		/*
		while(offset>0) {
		if(tid<offset) {
		if(tid+offset<threadNum)
		{
		shared[tid*3] += shared[(tid+offset)*3];
		shared[tid*3+1] += shared[(tid+offset)*3+1];
		shared[tid*3+2] += shared[(tid+offset)*3+2];
		}
		}
		threadNum=offset;
		if(offset==1)
		offset=0;
		else
		offset =(offset+1)/2;


		__syncthreads();
		}
		*/
		if(tid<2)
			shared[tid+1]=sqrt((float) shared[tid+1]);
		__syncthreads();
		if(tid == 0) {
			shared[1]=shared[1]*shared[2];
			dev_batchMatch[bidx*gridDim.y+bid] = (shared[0]/shared[1]+1)/2;  //写数据        
		}
	}	
}



void NNConfBatchcu(void ** batch,float *nccBatAnscu,int count,double &f2datacost)
{  	//输入时指针数组，输出时一个指针，指向一块内存
	//首先要把指针数组所有指向的Mat存入一个纹理内存batch2D中,用个for循环
	for(int i=0;i<count;i++)
		cudaMemcpy2D((char*)dev_batch+i*batch_pitch*patch_size, batch_pitch, batch[i], sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	//下面设计gpu网格,和线程块的维度
	//我们可以这样，给网格设定一个二维的dim3量（x,y),x行的线程块处理batch的第x*patch_size行到(x+1)*patch_size行的数据，即第x个小图
	//对于核函数，每个网格输出一个结果，所以我们先得在显存中申请一块内存用来存储结果,已经在nccinit中申请了dev_batchMatch
	dim3 blocks(count,pEx_size+nEx_size);
	if((patch_size*patch_size)>maxThreads)
		ThreadNUM=maxThreads;
	else
		ThreadNUM=patch_size*patch_size;//这样绝对不会出现多余的线程了
	dim3 threads(ThreadNUM);
	NNConfBatchker<<< blocks, threads,ThreadNUM*sizeof(float)*3>>>(dev_batchMatch,pEx_size);//一直在找错误，原来这里。。。无语了，复制代码很容易错
	cudaThreadSynchronize();
	cudaMemcpy(nccBatAnscu,dev_batchMatch,(pEx_size+nEx_size)*count*sizeof(float),cudaMemcpyDeviceToHost);	
    //获得从开始计时到停止之间的时间
  	f2datacost = 0;
}
__global__ void testDataker(float *dev_out)
{
	//for(int i=0;i<15;i++)
	//	dev_out[i]=tex2D(testData2D,i,0);
	int tid=threadIdx.x;//blockDim.x=patch_size
	int bid=blockIdx.y;
	int bidx=blockIdx.x;
	int threadNum=blockDim.x;
	int photox;;//第photox行
	int photoy;;//第photo列
	photox=tid/patch[0];
	photoy=tid%patch[0];
	if(blockIdx.x==9)
		if(tid==8)
			for(int i=0;i<6;i++)
				dev_out[i]=tex2D(batch2D,photoy+i,photox+bidx*patch[0]);
}
void testDatacu(const float *in,float *pout,float *nout)
{
	//in是从cpu传来的指针，pout,nout传出到cpu的数据
	//分配dev_testData，并且绑定为纹理
	cudaMallocPitch((void**)(&dev_testData), &testData_pitch, sizeof(float) * patch_size, patch_size*100);//程序里是float，所以这里也只处理float类型的。
	cudaChannelFormatDesc testDataDesc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, testData2D, dev_testData, testDataDesc, patch_size, patch_size*100, testData_pitch);
	cudaMemcpy2D((float*)((char*)dev_testData), testData_pitch, in, sizeof(float) * patch_size, sizeof(float) * patch_size, patch_size, cudaMemcpyHostToDevice);	
	//申请dev_out显存
	float *dev_out;
	cudaMalloc(&dev_out,15*sizeof(float));
	dim3 blocks(12,pEx_size+nEx_size);
	if((patch_size*patch_size)>512)
		ThreadNUM=512;
	else
		ThreadNUM=patch_size*patch_size;//这样绝对不会出现多余的线程了
	dim3 threads(ThreadNUM);
	testDataker<<<blocks, threads,ThreadNUM*sizeof(float)*3>>>(dev_out);
	cudaMemcpy(pout,dev_out,15*sizeof(float),cudaMemcpyDeviceToHost);
	//cudaMemcpy2D((float*)pout, 60, dev_pEx, pEx_pitch, sizeof(float) * patch_size, patch_size, cudaMemcpyDeviceToHost);	
	cudaMemcpy(nout,dev_batchMatch,(pEx_size+nEx_size)*12*sizeof(float),cudaMemcpyDeviceToHost);
	//cudaMemcpy2D((float*)nout, 60, dev_batch, batch_pitch, sizeof(float) * patch_size, patch_size*4, cudaMemcpyDeviceToHost);		
}
