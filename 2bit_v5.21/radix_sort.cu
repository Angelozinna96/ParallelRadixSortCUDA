#include <cstdio>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime_api.h>

#define restrict __restrict__

using namespace std;

int error(const char *msg) {
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

void cuda_check(cudaError_t err, const char *msg)
{
	if (err != cudaSuccess) { 
		fprintf(stderr, "%s: errore %d - %s\n",
			msg, err, cudaGetErrorString(err));
		exit(1);
	}
}
//inizializzazione su CPU con numeri random
void init_random(int vett[],int nels,int max)
{
	srand(time(NULL));
	for(int i=0;i<nels;++i)
		vett[i]=rand()% max+ 1;
}
//verifica con numeri random
bool verify_random(const int* scan_out, int nels)
{
	for(int i=0;i<nels-1;++i)
		if(scan_out[i]>scan_out[i+1])
		{
			fprintf(stderr, "errore tra le posizioni %d e %d \n", i,i+1);
			return false;
		}
		return true;
}
//verifica con numeri ordinati al contrario partendo da nels fino ad arrivare ad 1
bool verify(const int* scan_out, int nels)
{
	int err=0;
	for (int i = 0; i < nels; ++i) {
		if(i+1!=scan_out[i])
		{
			fprintf(stderr, "verify,idx=%d: val_scan:%d \n", i,scan_out[i]);
			err=1;
		}
	}
	if(err)
		return false;

	return true;
}
//inizializzazione su GPU
__global__ void init(int *vec, int nels)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < nels)
		vec[idx] = nels-idx;
}

extern __shared__ int4 shared[];

__device__ void scan_delle_code(int4 coda)
{
	__syncthreads();
	shared[threadIdx.x] = coda;
	for (int offset = 1; offset < blockDim.x; offset *= 2) {
		__syncthreads();
		if (threadIdx.x >= offset)
		{
			coda.x += shared[threadIdx.x - offset].x;
			coda.y += shared[threadIdx.x - offset].y;
			coda.z += shared[threadIdx.x - offset].z;
			coda.w += shared[threadIdx.x - offset].w;
		}
		__syncthreads();
		shared[threadIdx.x] = coda;
	}
	__syncthreads();
}
//primo scan
__global__ void scan_step1(int4 * restrict out0,int4 * restrict out1,int4 * restrict out2,int4 * restrict out3, const int4 * restrict in, int nels /* numero di quartine */, int * restrict code0,int * restrict code1,int * restrict code2,int * restrict code3, int nbit)
{
	int els_per_sezione = (nels + gridDim.x - 1)/gridDim.x;

	int idx = threadIdx.x + blockIdx.x*els_per_sezione;
	int4 val,val0,val1,val2,val3;
	int4 correzione_dal_blocco_precedente = make_int4(0,0,0,0);
	int numero_cicli = (els_per_sezione + blockDim.x - 1)/blockDim.x;
	int elemento_limite = min(els_per_sezione*(blockIdx.x+1), nels);

	for (int ciclo = 0; ciclo < numero_cicli;++ciclo, idx += blockDim.x) 
	{
		val = (idx < elemento_limite ? in[idx] : make_int4(0, 0, 0, 0));
		val0= make_int4(0,0,0,0);
		val1= make_int4(0,0,0,0);
		val2= make_int4(0,0,0,0);
		val3= make_int4(0,0,0,0);

		//basta fare solo 3 controlli sui bit,il quarto cofronto (00 per comodità) è complementare!
		//controllo sul primo valore della quartina (x)
		if(((val.x>>nbit)&3)==1)
			val1.x=1;

		else if(((val.x>>nbit)&3)==2)
			val2.x=1;

		else if(((val.x>>nbit)&3)==3)
			val3.x=1;

		else
			val0.x=1;

		//controllo sulla seconda componente della quartina (y)
		if(((val.y>>nbit)&3)==1)
			val1.y=1;
		
		else if(((val.y>>nbit)&3)==2)
			val2.y=1;
		
		else if(((val.y>>nbit)&3)==3)
			val3.y=1;

		else
			val0.y=1;
		//controllo sulla terza componente della quartina (z)
		if(((val.z>>nbit)&3)==1)
			val1.z=1;
		
		else if(((val.z>>nbit)&3)==2)
			val2.z=1;
		
		else if(((val.z>>nbit)&3)==3)
			val3.z=1;

		else
			val0.z=1;
		//controllo sulla quarta componente della quartina (w)
		if(((val.w>>nbit)&3)==1)
			val1.w=1;
		
		else if(((val.w>>nbit)&3)==2)
			val2.w=1;
		
		else if(((val.w>>nbit)&3)==3)
			val3.w=1;

		else 
			val0.w=1;
	
		/* scan delle componenti dei val */

		val0.y += val0.x;
		val0.z += val0.y;
		val0.w += val0.z;

		val1.y += val1.x;
		val1.z += val1.y;
		val1.w += val1.z;

		val2.y += val2.x;
		val2.z += val2.y;
		val2.w += val2.z;

		val3.y += val3.x;
		val3.z += val3.y;
		val3.w += val3.z;

		int4 coda=make_int4(val0.w,val1.w,val2.w,val3.w);
		scan_delle_code(coda);  

		int4 correzione_dai_thread_precedenti = make_int4(0,0,0,0);
		if (threadIdx.x > 0)
			correzione_dai_thread_precedenti = shared[threadIdx.x-1];

		int correzione_totale = correzione_dal_blocco_precedente.x + correzione_dai_thread_precedenti.x;
		val0.x += correzione_totale;
		val0.y += correzione_totale;
		val0.z += correzione_totale;
		val0.w += correzione_totale;

		correzione_totale = correzione_dal_blocco_precedente.y + correzione_dai_thread_precedenti.y;
		val1.x += correzione_totale;
		val1.y += correzione_totale;
		val1.z += correzione_totale;
		val1.w += correzione_totale;
		
		correzione_totale = correzione_dal_blocco_precedente.z + correzione_dai_thread_precedenti.z;
		val2.x += correzione_totale;
		val2.y += correzione_totale;
		val2.z += correzione_totale;
		val2.w += correzione_totale;
		
		correzione_totale = correzione_dal_blocco_precedente.w + correzione_dai_thread_precedenti.w;
		val3.x += correzione_totale;
		val3.y += correzione_totale;
		val3.z += correzione_totale;
		val3.w += correzione_totale;

		correzione_dal_blocco_precedente.x += shared[blockDim.x-1].x; //correzione di 00 (0)
		correzione_dal_blocco_precedente.y += shared[blockDim.x-1].y; //correzione di 01 (1)
		correzione_dal_blocco_precedente.z += shared[blockDim.x-1].z; //correzione di 10 (2)
		correzione_dal_blocco_precedente.w += shared[blockDim.x-1].w; //correzione di 11 (3)

		if (idx < nels)
		{
			out0[idx] = val0;
			out1[idx] = val1;
			out2[idx] = val2;
			out3[idx] = val3;	
		}
			
	}
	if (gridDim.x > 1 && threadIdx.x == blockDim.x - 1)
	{
		code0[blockIdx.x] = val0.w;
		code1[blockIdx.x] = val1.w;
		code2[blockIdx.x] = val2.w;
		code3[blockIdx.x] = val3.w;
	}
		
}
//secondo scan fatto sulo sui vettori di code
__global__ void scan_step2(int4 * restrict code0, int4 * restrict code1, int4 * restrict code2, int4 * restrict code3, int nels /* numero di quartine */)
{
	int els_per_sezione = (nels + gridDim.x - 1)/gridDim.x;
	int idx = threadIdx.x + blockIdx.x*els_per_sezione;
	int4 val0,val1,val2,val3;
	int4 correzione_dal_blocco_precedente = make_int4(0,0,0,0);
	int numero_cicli = (els_per_sezione + blockDim.x - 1)/blockDim.x;
	int elemento_limite = min(els_per_sezione*(blockIdx.x+1), nels);
	for (int ciclo = 0; ciclo < numero_cicli;++ciclo, idx += blockDim.x) 
	{
		val0 = (idx < elemento_limite ? code0[idx] : make_int4(0, 0, 0, 0));
		val1 = (idx < elemento_limite ? code1[idx] : make_int4(0, 0, 0, 0));
		val2 = (idx < elemento_limite ? code2[idx] : make_int4(0, 0, 0, 0));
		val3 = (idx < elemento_limite ? code3[idx] : make_int4(0, 0, 0, 0));

		/* scan delle componenti di val */

		val0.y += val0.x;
		val0.z += val0.y;
		val0.w += val0.z;

		val1.y += val1.x;
		val1.z += val1.y;
		val1.w += val1.z;

		val2.y += val2.x;
		val2.z += val2.y;
		val2.w += val2.z;

		val3.y += val3.x;
		val3.z += val3.y;
		val3.w += val3.z;

		//da modificare anche scan_delle_code
		int4 coda=make_int4(val0.w,val1.w,val2.w,val3.w);
		scan_delle_code(coda);  

		int4 correzione_dai_thread_precedenti = make_int4(0,0,0,0);
		if (threadIdx.x > 0)
			correzione_dai_thread_precedenti =
				shared[threadIdx.x-1];

		
		int correzione_totale = correzione_dal_blocco_precedente.x + correzione_dai_thread_precedenti.x;
		val0.x += correzione_totale;
		val0.y += correzione_totale;
		val0.z += correzione_totale;
		val0.w += correzione_totale;

		correzione_totale = correzione_dal_blocco_precedente.y + correzione_dai_thread_precedenti.y;
		val1.x += correzione_totale;
		val1.y += correzione_totale;
		val1.z += correzione_totale;
		val1.w += correzione_totale;
		
		correzione_totale = correzione_dal_blocco_precedente.z + correzione_dai_thread_precedenti.z;
		val2.x += correzione_totale;
		val2.y += correzione_totale;
		val2.z += correzione_totale;
		val2.w += correzione_totale;
		
		correzione_totale = correzione_dal_blocco_precedente.w + correzione_dai_thread_precedenti.w;
		val3.x += correzione_totale;
		val3.y += correzione_totale;
		val3.z += correzione_totale;
		val3.w += correzione_totale;

		correzione_dal_blocco_precedente.x += shared[blockDim.x-1].x; //correzione di 00 (0)
		correzione_dal_blocco_precedente.y += shared[blockDim.x-1].y; //correzione di 01 (1)
		correzione_dal_blocco_precedente.z += shared[blockDim.x-1].z; //correzione di 10 (2)
		correzione_dal_blocco_precedente.w += shared[blockDim.x-1].w; //correzione di 11 (3)

		if (idx < nels)
		{
			code0[idx] = val0;
			code1[idx] = val1;
			code2[idx] = val2;
			code3[idx] = val3;
		}
	}
}


__global__ void fixup(int4 * restrict scan0,int4 * restrict scan1,int4 * restrict scan2,int4 * restrict scan3,int nels ,const int * restrict code0,const int * restrict code1,const int * restrict code2,const int * restrict code3,const int4* restrict in,int nbit,int4* max)
{

	int els_per_sezione = (nels + gridDim.x - 1)/gridDim.x;

	int idx = threadIdx.x + blockIdx.x*els_per_sezione;

	int4 correzione_dal_blocco_precedente =make_int4(0,0,0,0);
	if(blockIdx.x>0)
	{
		correzione_dal_blocco_precedente.x = code0[blockIdx.x - 1];
		correzione_dal_blocco_precedente.y = code1[blockIdx.x - 1];
		correzione_dal_blocco_precedente.z = code2[blockIdx.x - 1];
		correzione_dal_blocco_precedente.w = code3[blockIdx.x - 1];
	}
	int numero_cicli = (els_per_sezione + blockDim.x - 1)/blockDim.x;
	int elemento_limite = min(els_per_sezione*(blockIdx.x+1), nels);

	for (int ciclo = 0; ciclo < numero_cicli;++ciclo, idx += blockDim.x) {
		if (idx < elemento_limite) 
		{
			int4 val0 = scan0[idx], val1=scan1[idx], val2=scan2[idx], val3=scan3[idx];
			int4 val_in=in[idx];
			if(idx==nels-1)
			{	//salvataggio in memoria globale degli ultimi elementi degli scan inclusivi
				(*max).x=val0.w+correzione_dal_blocco_precedente.x;
				(*max).y=val1.w+correzione_dal_blocco_precedente.y;
				(*max).z=val2.w+correzione_dal_blocco_precedente.z;
				(*max).w=val3.w+correzione_dal_blocco_precedente.w;
			}
			//trasformazione degli scan da inclusivi ad esclusivi
		
			val0.x += correzione_dal_blocco_precedente.x - ((((val_in.x>>nbit)&3)==0)?1:0);
			val0.y += correzione_dal_blocco_precedente.x - ((((val_in.y>>nbit)&3)==0)?1:0);
			val0.z += correzione_dal_blocco_precedente.x - ((((val_in.z>>nbit)&3)==0)?1:0);
			val0.w += correzione_dal_blocco_precedente.x - ((((val_in.w>>nbit)&3)==0)?1:0);

			val1.x += correzione_dal_blocco_precedente.y - ((((val_in.x>>nbit)&3)==1)?1:0);
			val1.y += correzione_dal_blocco_precedente.y - ((((val_in.y>>nbit)&3)==1)?1:0);
			val1.z += correzione_dal_blocco_precedente.y - ((((val_in.z>>nbit)&3)==1)?1:0);
			val1.w += correzione_dal_blocco_precedente.y - ((((val_in.w>>nbit)&3)==1)?1:0);

			val2.x += correzione_dal_blocco_precedente.z - ((((val_in.x>>nbit)&3)==2)?1:0);
			val2.y += correzione_dal_blocco_precedente.z - ((((val_in.y>>nbit)&3)==2)?1:0);
			val2.z += correzione_dal_blocco_precedente.z - ((((val_in.z>>nbit)&3)==2)?1:0);
			val2.w += correzione_dal_blocco_precedente.z - ((((val_in.w>>nbit)&3)==2)?1:0);

			val3.x += correzione_dal_blocco_precedente.w - ((((val_in.x>>nbit)&3)==3)?1:0);
			val3.y += correzione_dal_blocco_precedente.w - ((((val_in.y>>nbit)&3)==3)?1:0);
			val3.z += correzione_dal_blocco_precedente.w - ((((val_in.z>>nbit)&3)==3)?1:0);
			val3.w += correzione_dal_blocco_precedente.w - ((((val_in.w>>nbit)&3)==3)?1:0);
			
			scan0[idx] = val0;
			scan1[idx] = val1;
			scan2[idx] = val2;
			scan3[idx] = val3;
		}
	}
}
//kernel adibito al riordino dei vettori utilizzando 2 bit
__global__ void reorder(const int4* restrict scan0,const int4* restrict scan1,const int4* restrict scan2,const int4* restrict scan3,const int4* restrict in, int* restrict out,int nels,int4* max)
{
		int els_per_sezione = (nels + gridDim.x - 1)/gridDim.x;
		int idx = threadIdx.x + blockIdx.x*els_per_sezione;
		int numero_cicli = (els_per_sezione +blockDim.x -1)/blockDim.x;
		int elemento_limite = min(els_per_sezione*(blockIdx.x+1), nels);
		int4 offset_max=make_int4((*max).x,(*max).y,(*max).z,(*max).w);	
		int4 val_scan_succ;
		//scan dei max
		int4 max_scan=make_int4(offset_max.x,offset_max.y,offset_max.z,offset_max.w);
		offset_max.x=0;    						//offset di 00 (0)
		offset_max.y=max_scan.x;				//offset di 01 (1)
		offset_max.z=offset_max.y + max_scan.y; //offset di 10 (2)
		offset_max.w=offset_max.z + max_scan.z; //offset di 11 (3)

		//inizio ciclo di reorder
		for (int ciclo = 0; ciclo < numero_cicli;++ciclo, idx += blockDim.x) 
		{
			if (idx < elemento_limite) 
			{
				int4 val_num=in[idx], val_scan0=scan0[idx], val_scan1=scan1[idx], val_scan2=scan2[idx], val_scan3=scan3[idx];
				 // confronto 1° elemento con 2° elemento della quartina
				if(val_scan0.x!=val_scan0.y) 
					out[val_scan0.x+offset_max.x]=val_num.x;

				else if(val_scan1.x!=val_scan1.y)
					out[val_scan1.x+offset_max.y]=val_num.x;

				else if(val_scan2.x!=val_scan2.y)
					out[val_scan2.x+offset_max.z]=val_num.x;
				else
					out[val_scan3.x+offset_max.w]=val_num.x;

				// confronto 2° elemento con 3° elemento della quartina
				if(val_scan0.y!=val_scan0.z) 
					out[val_scan0.y+offset_max.x]=val_num.y;

				else if(val_scan1.y!=val_scan1.z)
					out[val_scan1.y+offset_max.y]=val_num.y;

				else if(val_scan2.y!=val_scan2.z)
					out[val_scan2.y+offset_max.z]=val_num.y;
				else
					out[val_scan3.y+offset_max.w]=val_num.y;

				// confronto 3° elemento con 4° elemento della quartina
				if(val_scan0.z!=val_scan0.w) 
					out[val_scan0.z+offset_max.x]=val_num.z;

				else if(val_scan1.z!=val_scan1.w)
					out[val_scan1.z+offset_max.y]=val_num.z;

				else if(val_scan2.z!=val_scan2.w)
					out[val_scan2.z+offset_max.z]=val_num.z;
				else
					out[val_scan3.z+offset_max.w]=val_num.z;	

				//confronto 4° elemento con 1° elemento della quartina dello scan successivo

				if(idx!=nels-1)
				{
					//scan3[idx+1] puo non essere preso, visto che non viene mai usato
					val_scan_succ=make_int4(scan0[idx+1].x,scan1[idx+1].x,scan2[idx+1].x,0); // primi valori delle quartine successive degli scan

					if(val_scan0.w!=val_scan_succ.x) 
						out[val_scan0.w + offset_max.x]=val_num.w;

					else if(val_scan1.w!=val_scan_succ.y)
						out[val_scan1.w + offset_max.y]=val_num.w;

					else if(val_scan2.w!=val_scan_succ.z)
						out[val_scan2.w + offset_max.z]=val_num.w;
					else
						out[val_scan3.w + offset_max.w]=val_num.w;	

				}
				else
				{

					if(val_scan0.w!=max_scan.x) 
						out[val_scan0.w + offset_max.x]=val_num.w;

					else if(val_scan1.w!=max_scan.y)
						out[val_scan1.w + offset_max.y]=val_num.w;

					else if(val_scan2.w!=max_scan.z)
						out[val_scan2.w + offset_max.z]=val_num.w;
					else
						out[val_scan3.w + offset_max.w]=val_num.w;	
				}
			}
		}



}


int main(int argc, char *argv[])
{
	if (argc < 4)
		error("sintassi radix_sort: numels thread_per_blocco numero_blocchi_scan valore_massimo");

	int nels = atoi(argv[1]); /* numero di elementi */
	if (nels <= 0)
		error("il numero di elementi deve essere positivo");
	if (nels & 3)
		error("il numero di elementi deve essere multiplo di 4");

	int numThreads = atoi(argv[2]); /* local work size */
	if (numThreads <= 0)
		error("il numero di thread per blocco deve essere positivo");

	int numBlocksScan = atoi(argv[3]); /* numero blocchi scan */
	if (numBlocksScan <= 0)
		error("il numero di blocchi deve essere positivo");
	int numMax = atoi(argv[4]); /* numero blocchi scan */
	if (numMax <= 0)
		error("il valore massimo deve essere positivo");
	//inizializzazione dei vettori
	const size_t memsize = sizeof(int)*nels;
	int4 *d_v1, *d_scan0,*d_scan1,*d_scan2,*d_scan3, *d_code0,*d_code1,*d_code2,*d_code3,*d_out,*tmp;
	int numbit;
	int4 *d_max;
	//calolo dei cicli da fare avendo il massimo
	int cicli=int(log(numMax)/log(2)) + 1;
	printf("numero cicli da fare=%d\n",cicli/2);
	//allocazione dei vettori su GPU
	cudaError_t err = cudaMalloc(&d_v1, memsize);
	cuda_check(err, "malloc v1");
	err= cudaMalloc(&d_max,sizeof(int4));
	cuda_check(err,"malloc max");

	err = cudaMalloc(&d_scan0, memsize);
	cuda_check(err, "malloc scan0");
	err = cudaMalloc(&d_scan1, memsize);
	cuda_check(err, "malloc scan1");
	err = cudaMalloc(&d_scan2, memsize);
	cuda_check(err, "malloc scan2");
	err = cudaMalloc(&d_scan3, memsize);
	cuda_check(err, "malloc scan3");

	err = cudaMalloc(&d_code0, numBlocksScan*sizeof(int));
	cuda_check(err, "malloc code0");
	err = cudaMalloc(&d_code1, numBlocksScan*sizeof(int));
	cuda_check(err, "malloc code1");
	err = cudaMalloc(&d_code2, numBlocksScan*sizeof(int));
	cuda_check(err, "malloc code2");
	err = cudaMalloc(&d_code3, numBlocksScan*sizeof(int));
	cuda_check(err, "malloc code3");

	err = cudaMalloc(&d_out, memsize);
	cuda_check(err, "malloc out");
	//allocazione su CPU
	int *vout = (int*)malloc(memsize);
	if (!vout)
		error("alloc vscan");
	//inizializzazione su CPU di numeri random con massimo possibile numMax
	init_random(vout,nels,numMax);
	/*
	//inizializzazione su GPU di numeri decrescenti a partire da nels
	int numBlocks = (nels + numThreads - 1)/numThreads;
	init<<<numBlocks, numThreads>>>((int*)d_v1, nels);
	*/
	//prova ad otimizzare la cache
	cudaFuncSetCacheConfig(scan_step1,  cudaFuncCachePreferL1);	
	cudaFuncSetCacheConfig(scan_step2,  cudaFuncCachePreferL1);	
	cudaFuncSetCacheConfig(fixup,  cudaFuncCachePreferL1);	
	cudaFuncSetCacheConfig(reorder,  cudaFuncCachePreferL1);	
	
	err = cudaMemcpy(d_v1,vout,memsize, cudaMemcpyHostToDevice);
	cuda_check(err, "memcpy vett su GPU");
	//creazione eventi
	cudaEvent_t before_scan, after_scan;

	err = cudaEventCreate(&before_scan);
	cuda_check(err, "create event before");
	err = cudaEventCreate(&after_scan);
	cuda_check(err, "create event after");

	cudaEventRecord(before_scan);

	for(numbit=0;numbit<cicli;numbit+=2)
	{
		//pulizia dei vettori di code
		err= cudaMemset (d_code0,0, numBlocksScan*sizeof(int));
		cuda_check(err, "memset0");
		err= cudaMemset (d_code1,0, numBlocksScan*sizeof(int));
		cuda_check(err, "memset1");
		err= cudaMemset (d_code2,0, numBlocksScan*sizeof(int));
		cuda_check(err, "memset2");
		err= cudaMemset (d_code3,0, numBlocksScan*sizeof(int));
		cuda_check(err, "memset3");

		scan_step1<<<numBlocksScan, numThreads, numThreads*sizeof(int)*4>>>
			(d_scan0,d_scan1,d_scan2,d_scan3, d_v1, nels/4, (int*)d_code0,(int*)d_code1,(int*)d_code2,(int*)d_code3, numbit);

		scan_step2<<<1, numThreads, numThreads*sizeof(int)*4>>>
			(d_code0, d_code1,d_code2,d_code3, numBlocksScan/4);

		fixup<<<numBlocksScan, numThreads>>>(d_scan0,d_scan1,d_scan2,d_scan3, nels/4, (int*)d_code0,(int*)d_code1,(int*)d_code2,(int*)d_code3,d_v1,numbit,d_max);
		
		reorder<<<numBlocksScan,numThreads>>>(d_scan0,d_scan1,d_scan2,d_scan3,d_v1,(int*)d_out,nels/4,d_max);
		
		//scambio deii puntatori d_out e d_v1
		if(numbit+2 <cicli)
		{
			tmp=d_v1;
			d_v1=d_out;	
			d_out=tmp;
		}	
	}

	cudaEventRecord(after_scan);
	err = cudaEventSynchronize(after_scan);
	cuda_check(err, "after scan sznc");
	float runtime_ms;
	cudaEventElapsedTime(&runtime_ms, before_scan, after_scan);
	printf("scan runtime: %.4g ms\n", runtime_ms);
	err = cudaMemcpy(vout, d_out, memsize, cudaMemcpyDeviceToHost);
	cuda_check(err, "memcpy");
	printf("\n\n");

	verify_random(vout,nels)?printf("Ordinamento riuscito!\n"):printf("Ordinamento non riuscito!\n");
	
	if(nels <=32)
		for(int i=0;i<nels;++i)
			printf("%d ",vout[i]);
	printf("\n");


}
