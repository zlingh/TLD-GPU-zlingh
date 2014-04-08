#include <opencv2/opencv.hpp>
#include "tld_utils.h"
#include "iostream"
#include "sstream"
#include "tld.h"
#include "stdio.h"
#include <time.h>
//#include <Windows.h>
using namespace cv;
using namespace std;
#undef UNICODE // 如果你不知道什么意思，请不要修改
//Global variables
#include <stdlib.h>
#include <Windows.h>
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
bool fromCa=false;
long totalFrameNumber;
bool isImage=false;
char **imageList=NULL;
int listCount=0;
string video;
//如果不想显示控制台则取消下面的注释
#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" )
//这个函数返回directory这个目录下的所有文件名（包括后缀名）到一个指针数组中，指针数组中每一个是一个char型的指针，
//返回的是个指针，内容是这个指针数组的首地址，类型是char*，即一个4个字节的内存地址。*ans得到的是一个指向cha的指针。
char** EnumFiles(const char *directory, int *count)
{
	int direcLen=0;//directory长度
	while(*(directory+(++direcLen))!=0);
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;
	int resultSize=500;
//	char result[MAX_RESULT][MAX_PATH];
	char *result=(char*)malloc(resultSize*MAX_PATH);// 打算加一个动态申请内存的功能，既不浪费又够用
	char **returnresult;
	char pattern[MAX_PATH];
	int i = 0, j;
	// 开始查找
	strcpy(pattern, directory);
	strcat(pattern, "\\*.*");
	hFind = FindFirstFile(pattern, &FindFileData);

	if (hFind == INVALID_HANDLE_VALUE) 
	{
		*count = 0;
		return NULL;
	} 
	else 
	{
		do
		{
			if(i==resultSize)
			{ 
				char *temp=(char*)malloc(resultSize*2*MAX_PATH);
				memcpy(temp,result,resultSize*MAX_PATH*sizeof(char));//strcpy(temp,result);//哦这里只复制了\0前面的
				char *temp1=result;
				result=temp;
				free(temp1);
				resultSize*=2;
			}
			strcpy(result+i*MAX_PATH,directory);
			*(result+(i)*MAX_PATH+direcLen)='\\';
			strcpy(result+(i++)*MAX_PATH+direcLen+1, FindFileData.cFileName);
		}
		while (FindNextFile(hFind, &FindFileData) != 0);
	}

	// 查找结束
	FindClose(hFind);

	// 复制到结果中
	returnresult = (char **)calloc(i, sizeof(char *));

	for (j = 0; j < i-2; j++)
	{
		returnresult[j] = (char *)calloc(MAX_PATH, sizeof(char));
		strcpy(returnresult[j], result+(j+2)*MAX_PATH);
	}

	*count = i-2;
	return returnresult;
}

void readBB(char* file){
	ifstream bb_file (file);
	string line;
	getline(bb_file,line);
	istringstream linestream(line);
	string x1,y1,x2,y2;
	getline (linestream,x1, ',');
	getline (linestream,y1, ',');
	getline (linestream,x2, ',');
	getline (linestream,y2, ',');
	int x = atoi(x1.c_str());// = (int)file["bb_x"];
	int y = atoi(y1.c_str());// = (int)file["bb_y"];
	int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
	int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
	box = Rect(x,y,w,h);
}
//bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
	switch( event ){
	case CV_EVENT_MOUSEMOVE:
		if (drawing_box){
			box.width = x-box.x;
			box.height = y-box.y;
		}
		break;
	case CV_EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = Rect( x, y, 0, 0 );
		break;
	case CV_EVENT_LBUTTONUP:
		drawing_box = false;
		if( box.width < 0 ){
			box.x += box.width;
			box.width *= -1;
		}
		if( box.height < 0 ){
			box.y += box.height;
			box.height *= -1;
		}
		gotBB = true;
		break;
	}
}

void print_help(char** argv){
	printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
	printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}
void read_options(int len, char** c,VideoCapture& capture,FileStorage &fs){
		for (int i=0;i<len;i++){
		if (strcmp(c[i],"-b")==0&&!fromCa){
			if (len>i){
				readBB(c[i+1]);
				gotBB = true;
			}
			else
				print_help(c);
		}
		if (strcmp(c[i],"-s")==0&&!fromCa){
			if (len>i){
				video = string(c[i+1]);//continue;
				capture.open(video);
				fromfile = true;
			}
			else
				print_help(c);

		}
		if (strcmp(c[i],"-p")==0){
			if (len>i){
				fs.open(c[i+1], FileStorage::READ);
			}
			else
				print_help(c);
		}
		if (strcmp(c[i],"-no_tl")==0){
			tl = false;
		}
		if (strcmp(c[i],"-r")==0){
			rep = true;
		}
		if(strcmp(c[i],"-im")==0){
			char *directory=c[i+1];
			imageList=EnumFiles(directory,&listCount);
			char *temp=new char[20];
			for(int i=0;i<20;i++)
			{
				if(imageList[0][i]!='\\')
				temp[i]=imageList[0][i];
				else
				{
					temp[i]='\\';
					temp[i+1]=0;
					break;
				}
			}
			//listCount=listCount-2;//第一个是.，第二个是..，去掉不算。
		    isImage=true;
			fromfile = true;
			fromCa=false;
			temp=strcat(temp,"init.txt");
			if(strcmp(imageList[listCount-1],temp)==0)
			{
				listCount--;
				readBB(imageList[listCount]);
				gotBB = true;
			}
		}
	}
}

void read_optionsbackup(int argc, char** argv,VideoCapture& capture,FileStorage &fs){
	for (int i=0;i<argc;i++){
		if (strcmp(argv[i],"-b")==0&&!fromCa){
			if (argc>i){
				readBB(argv[i+1]);
				gotBB = true;
			}
			else
				print_help(argv);
		}
		if (strcmp(argv[i],"-s")==0&&!fromCa){
			if (argc>i){
				video = string(argv[i+1]);
				capture.open(video);
				fromfile = true;
			}
			else
				print_help(argv);

		}
		if (strcmp(argv[i],"-p")==0){
			if (argc>i){
				fs.open(argv[i+1], FileStorage::READ);
			}
			else
				print_help(argv);
		}
		if (strcmp(argv[i],"-no_tl")==0){
			tl = false;
		}
		if (strcmp(argv[i],"-r")==0){
			rep = true;
		}
	}
}
vector<string> splitEx(const string& src, string separate_character)
{
	vector<string> strs;

	int separate_characterLen = separate_character.size();//分割字符串的长度,这样就可以支持如“,,”多字符串的分隔符
	int lastPosition = 0,index = -1;
	while (-1 != (index = src.find(separate_character,lastPosition)))
	{
		strs.push_back(src.substr(lastPosition,index - lastPosition));
		lastPosition = index + separate_characterLen;
	}
	string lastString = src.substr(lastPosition);//截取最后一个分隔符后的内容
	if (!lastString.empty())
		strs.push_back(lastString);//如果最后一个分隔符后还有内容就入队
	return strs;
}
int main(int argc, char * argv[]){
	int patcharray[6]={15,20,25,30,35};
	int minwind[3]={5,10,15};
	FILE *pfilezp;//=fopen("Record.txt","w");
	FILE *objectf;
	FILE *tablef;
	FILE *patchf;
	time_t start,end;
	double wholecost;
	struct tm *ptr;
	int retry;
	int startFrame=0;
	bool nopoint=true;//是否显示点
	bool drawDec=false;//是否显示detection的框框
	bool cameraAgain=false;
	bool breaknow=false;//为了退出大循环所设的变量
	bool play=false;//是否切换到play模式	
	char *test[]={
		"-p parameters.yml -s car.mpg -b car.txt",
		"-p ../parameters.yml -s ../datasets/01_david/david.mpg -b ../datasets/01_david/init.txt",
		"-p ../parameters.yml -s ../datasets/02_jumping/jumping.mpg -b ../datasets/02_jumping/init.txt",
		"-p ../parameters.yml -s ../datasets/03_pedestrian1/pedestrian1.mpg -b ../datasets/03_pedestrian1/init.txt",
		"-p ../parameters.yml -s ../datasets/04_pedestrian2/pedestrian2.mpg -b ../datasets/04_pedestrian2/init.txt",
		"-p ../parameters.yml -s ../datasets/05_pedestrian3/pedestrian3.mpg -b ../datasets/05_pedestrian3/init.txt",
		"-p ../parameters.yml -s ../datasets/06_car/car.mpg -b ../datasets/06_car/init.txt",
		"-p ../parameters.yml -s ../datasets/07_motocross/motocross.mpg -b ../datasets/07_motocross/init.txt",
		//"-p ../parameters.yml -s ../datasets/08_volkswagen/volkswagen.mpg -b ../datasets/08_volkswagen/init.txt",
		"-p ../parameters.yml -s ../datasets/09_carchase/carchase.mpg -b ../datasets/09_carchase/init.txt",
		"-p ../parameters.yml -s ../datasets/10_panda/panda.mpg -b ../datasets/10_panda/init.txt",
		"-p ../parameters.yml -s ../datasets/11_test/test2.avi"};
	char *testt[]={"-p parameters.yml -im data"};//,"-p parameters.yml -s car.mpg -b init1.txt",
		//"-p parameters.yml -s test.avi",
	//	"-p parameters.yml -s motocross.mpg -b init2.txt"};
	for(int i=0;i<1;i++){
		for (int flag=0;flag<1;flag++)
		//for (int pi=0;pi<15;pi++)		
		{
			RNG RNG( int64 seed=-1 );
			double costsum[7]={0.0,0.0,0.0,0.0,0.0,0.0,0.0};
			if(flag==1)
				int tempp=1;
			isImage=false;
			breaknow=false;
			retry=-1;
			patchf=fopen("patchgpu.txt", "at");
			pfilezp=fopen("Record.txt","at");
			tablef=fopen("tableout.txt","at");
			objectf=fopen("objectf.txt", "at");			
			drawing_box = false;
			gotBB = false;
			tl = true;
			rep = false;
			fromfile=false;
			start=time(NULL); ptr=localtime(&start);
			printf(asctime(ptr));
			fprintf(pfilezp,asctime(ptr));
			wholecost = (double)getTickCount();
			VideoCapture capture;
			//CvCapture* capture;
			capture.open(1);
			//capture = cvCaptureFromCAM( CV_CAP_ANY);
			FileStorage fs;
			//Read options
			string s = test[flag];
			string del = " ";
			char test2[10][100];
			test2[4][0]='0';//这里有很奇怪的事情，下次循环时竟然保留了上次循环的test2的值，按理说test2是在循环里面定义的，应该是个局部变量，每次循环应该是新开的变量啊。
			vector<string> strs = splitEx(s, del);
			for ( unsigned int i = 0; i < strs.size(); i++)
			{  
				//  cout << strs[i].c_str() << endl;
				//	test2[i]=strs[i].c_str();
				strcpy(test2[i],strs[i].c_str());
				//cout<<test2[i]<<endl;
			}
			//int tp=strs.size();
			char *p[10];
			char **test3;//搞不太懂这里啊。。。
			for(int i=0;i<10;i++)
				p[i]=test2[i];
			test3=p; 	

			read_options(10,test3,capture,fs);

//			video = string(argv[1]);//目标视频//实验中输入参数就是这三行
//			capture.open(video);
//			readBB(argv[2]);//目标框


			// read_options(argc,argv,capture,fs);
			if(startFrame>0)//说明按下了r键，要我们重新手动选择框框
			{				
				box = Rect( 0, 0, 0, 0 );
				gotBB=false;
			}
			//   read_options(argc,argv,capture,fs);
			//Init camera
			if (!capture.isOpened()&&!isImage)//打不开视频而且不是图像序列
			{
				cout << "capture device failed to open!" << endl;
				return 1;
			}
			//Register mouse callback to draw the bounding box
			cvNamedWindow("TLD",CV_WINDOW_AUTOSIZE);
			cvSetMouseCallback("TLD", mouseHandler, NULL);
			//TLD framework
			TLD tld;
			//Read parameters file
			tld.read(fs.getFirstTopLevelNode());
//			tld.patch_size=atoi(argv[3]);
//			tld.min_win=atoi(argv[4]);	
			Mat frame;
			Mat last_gray;
			Mat first;
			if(fromCa)
			fromfile=false;
			fromCa=false;
			if (fromfile){
				if(!isImage){
					//	capture >> frame;
					totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);  
					cout<<"整个视频共"<<totalFrameNumber<<"帧"<<endl;
					//	capture.set( CV_CAP_PROP_POS_FRAMES,0); 似乎没有用
					for(int i=0;i<=startFrame;i++){
						capture.read(frame);}
					cvtColor(frame, last_gray, CV_RGB2GRAY);
					frame.copyTo(first);
				}
				else{
					totalFrameNumber = listCount;  
					cout<<"整个图像序列共"<<listCount<<"帧"<<endl;
					//	capture.set( CV_CAP_PROP_POS_FRAMES,0); 似乎没有用
					frame=imread(imageList[startFrame+2]);
					cvtColor(frame, last_gray, CV_RGB2GRAY);
					frame.copyTo(first);
				}

			}else{
				capture.set(CV_CAP_PROP_FRAME_WIDTH,340);
				capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
			}

			///Initialization
GETBOUNDINGBOX:
			while(!gotBB)
			{
				if (!fromfile){
					capture >> frame;
				}
				else
					first.copyTo(frame);
				cvtColor(frame, last_gray, CV_RGB2GRAY);
				drawBox(frame,box);
				imshow("TLD", frame);
				int cw=cvWaitKey(1);
				if (cw == 'q')
					return 0;
				if(cw=='p')
				{
					play=true;box=Rect( 0, 0, 15, 15 );
					break;
				}
			}
			if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"]){
				cout << "Bounding box too small, try again." << endl;
				gotBB = false;
				goto GETBOUNDINGBOX;
			}
			//Remove callback
			cvSetMouseCallback( "TLD", NULL, NULL );
			printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);

			//Output file
			FILE  *bb_file = fopen("bounding_boxes.txt","w");
			//fprintf(tablef,"%s\n",test2[3]);
			//TLD initialization
			tld.initNcc();
			tld.init(last_gray,box,bb_file);
			tld.initGpu(last_gray);
			///Run-time
			Mat current_gray;
			BoundingBox pbox;
			vector<Point2f> pts1;
			vector<Point2f> pts2;
			bool status=true;
			int frames = 1;
			int detections = 1;
			int flagg=startFrame;//记录是第几帧

			// pfilezp=fopen("Record.txt","w");
REPEAT:     
			//	capture.set( CV_CAP_PROP_POS_FRAMES,startFrame);  			
			while((!isImage&&capture.read(frame))||(isImage)){
				if(isImage){					
					frame=imread(imageList[startFrame++]);
					if(startFrame>listCount-1){
						box=Rect( 0, 0, 0, 0 );
						break;}
				}
				
				flagg++;
				double cost = (double)getTickCount();
				//get frame
				cvtColor(frame, current_gray, CV_RGB2GRAY);
				//Process Frame  				
				if(!play)
					tld.processFrame(last_gray,current_gray,pts1,pts2,pbox,status,tl,bb_file,tablef,costsum,objectf);  
				//Draw Points
				if (status&&!play){
					if(!nopoint){
						drawPoints(frame,pts1);
						drawPoints(frame,pts2,Scalar(0,255,0));
					}
					drawBox(frame,pbox,Scalar(255,255,255),2);
					detections++;
				}
				if(drawDec){
				//	for(int j=0;j<tld.dt.bb.size();j++)
				//		drawBox(frame,tld.grid[tld.dt.bb[j]]);
					for(int j=0;j<tld.dbb.size();j++)//此处为了论文中图片所写，才把dbb的显示出来，因为使用孤立点算法以后会存入dbb。
						drawBox(frame,tld.dbb[j]);
				}
				//Display
				imshow("TLD", frame);
				//swap points and images
				swap(last_gray,current_gray);
				pts1.clear();
				pts2.clear();
				frames++;
				if(frames==tld.pausenum) system("pause");				
				printf("Detection rate: %d/%d\n",detections,frames);
				if(frames==totalFrameNumber)
					tld.islast=true;
				cost=getTickCount()-cost;
				printf("--------------------------------process cost %gms\n", cost*1000/getTickFrequency());
				int c = waitKey(1);
				//int c= 'm';
				if(cameraAgain)
					c='c';//如果在camera模式下按下了r键则会回到这里。
				switch(c){
				case 'n'://测试下一段视频
					{				
					//	retry==1;
						breaknow=true; 
						gotBB=false;
						box = Rect(0, 0, 0, 0);
						break;
					}
				case 'q':{
					tld.endGpu();
					return 0;}
				case 'r'://要手动在当前帧重新选择目标框框
					{
						if(fromfile)
						{
						if(play)
						startFrame=flagg;
						else
							startFrame=flagg-1;
						play=false;
						flag--;
						retry=1;						
						breaknow=true;
						break;
						}
						else{//如果摄像头模式时按了r想重新选择目标框框键，则相当于想再次打开Cam模式
						cameraAgain=true;
						break;
						}
					}		
				case 'x':
					{
						nopoint=!nopoint;
						break;
					}
				case 'd':
					{
						drawDec=!drawDec;
						break;
					}
				case 'p':
					{
						play=!play;
						break;
					}
				case 's':
					{
						//想用来打开或者关闭控制台
						//#pragma comment( linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"" )
					}
				case 'c'://切换到使用摄像头模式
					{
						cameraAgain=false;
						breaknow=true;
						fromCa=true;//readoptions函数里会用到这个判断，如果fromCa为true，即使有文件也不读相当于只是输入"-p ../parameters.yml"
						retry=1;
						box=Rect( 0, 0, 0, 0 );
						flag--;//循环倒退回去,但是不能
						break;
					}

				}
				if(breaknow) break;
				// if(flagg>=9)
				// fclose(pfilezp);
			} 
			tld.endGpu();
			fprintf(pfilezp,"num=%d %s\n",flag,test2[3]);
			fprintf(pfilezp,"patch_size=%d\n",tld.patch_size);
			fprintf(pfilezp,"USE GPU OR CPU：%s\n",tld.use);
			fprintf(pfilezp,"Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
			fprintf(pfilezp,"Detection rate: %d/%d\n",detections,frames);
			fprintf(pfilezp,"classifier.pEx: %d classifier.nEx: %d\n", tld.classifier.pEx.size(),tld.classifier.nEx.size());
			fclose(bb_file);
			wholecost=(getTickCount()-wholecost);
			end=time(NULL);
			fprintf(pfilezp,"timecost = %gms \n",wholecost*1000/getTickFrequency());
			fprintf(pfilezp,"every frame cost %g ms \n",wholecost*1000/getTickFrequency()/frames);
			fprintf(pfilezp,"%gms %gms %gms\n\n",costsum[0]/frames,costsum[1]/frames,costsum[2]/frames);
			//依次打印视频名字/检测率/patch宽/patch高/总时间/filter1/filter2/detect/track/learn/filter1数据拷贝时间/filter2数据拷贝时间
			fprintf(patchf,"%s\t %d/%d\t%d\t%d\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n",argv[1],detections,frames,tld.patch_size,tld.min_win,wholecost*1000/getTickFrequency()/frames,costsum[0]/frames,costsum[1]/frames,costsum[2]/frames,costsum[3]/frames,costsum[4]/frames,costsum[5]/frames,costsum[6]/frames);
			//time_t start,end;
			//start=time(NULL); ptr=localtime(&start); printf(asctime(ptr));	 
			//fprintf(pfilezp,"timecost2=%f ms\n",difftime(end,start)*1000);
			fclose(pfilezp);	
			fclose(tablef);
			fclose(patchf);
			if(retry==1)
			{
				continue;
			}//startFrame不归零
			startFrame=0;//重置startFrame			
			if (rep){
				rep = false;
				tl = false;
				fclose(bb_file);
				bb_file = fopen("final_detector.txt","w");
				//capture.set(CV_CAP_PROP_POS_AVI_RATIO,0);
				capture.release();
				capture.open(video);
				goto REPEAT;
			} 
		}
	}
	//  fclose(pfilezp);
	// fclose(pfilezpp);
	//char c=getchar();
	return 0;
}