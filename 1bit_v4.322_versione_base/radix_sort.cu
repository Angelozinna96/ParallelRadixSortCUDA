#include <cstdio>
#include <math.h>
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
//init su GPU di Numeri ordinati al contrario
__global__ void init(int *vec, int nels)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if (idx < nels)
		vec[idx] = nels-idx;
}

extern __shared__ int shared[];

__device__ void scan_delle_code(int coda)
{
	__syncthreads();
	shared[threadIdx.x] = coda;
	for (int offset = 1; offset < blockDim.x; offset *= 2) {
		__syncthreads();
		if (threadIdx.x >= offset)
			coda += shared[threadIdx.x - offset];
		__syncthreads();
		shared[threadIdx.x] = coda;
	}
	__syncthreads();
}

__global__ void scan_step1(int4 * restrict out, const int4 * restrict in, int nels /* numero di quartine */, int * restrict code, int nbit,int flag_primo_lancio,int shift)
{
	int els_per_sezione = (nels + gridDim.x - 1)/gridDim.x;

	int idx = threadIdx.x + blockIdx.x*els_per_sezione;

	int4 val;
	int correzione_dal_blocco_precedente = 0;
	if(blockIdx.x>0)
		correzione_dal_blocco_precedente  = code[blockIdx.x - 1];


	int numero_cicli = (els_per_sezione + blockDim.x - 1)/blockDim.x;
	int elemento_limite = min(els_per_sezione*(blockIdx.x+1), nels);

	for (int ciclo = 0; ciclo < numero_cicli;++ciclo, idx += blockDim.x) 
	{
		val = (idx < elemento_limite ? in[idx] : make_int4(0, 0, 0, 0));

		if(flag_primo_lancio) //primo lancio di scan_step
		{
			//complemento gia fatto invertendo le assegnazioni	
			val.x =((val.x)&shift)?0:1; 
			val.y =((val.y)&shift)?0:1;
			val.z =((val.z)&shift)?0:1;
			val.w =((val.w)&shift)?0:1;
		}

		/* scan delle componenti di val */
		val.y += val.x;
		val.z += val.y;
		val.w += val.z;

		scan_delle_code(val.w);

		int correzione_dai_thread_precedenti = 0;
		if (threadIdx.x > 0)
			correzione_dai_thread_precedenti =
				shared[threadIdx.x-1];

		int correzione_totale = correzione_dal_blocco_precedente + correzione_dai_thread_precedenti;
		val.x += correzione_totale;
		val.y += correzione_totale;
		val.z += correzione_totale;
		val.w += correzione_totale;

		correzione_dal_blocco_precedente += shared[blockDim.x-1];

		if (idx < nels)
			out[idx] = val;
	}
	if (gridDim.x > 1 && threadIdx.x == blockDim.x - 1)
		code[blockIdx.x] = val.w;
}

__global__ void fixup(int4 * restrict scan,int nels /* numero di quartine */,const int * restrict code,const int4* restrict in,int nbit,int* max,int shift)
{

	int els_per_sezione = (nels + gridDim.x - 1)/gridDim.x;

	int idx = threadIdx.x + blockIdx.x*els_per_sezione;

	int correzione_dal_blocco_precedente =0;
	if(blockIdx.x>0)
		correzione_dal_blocco_precedente = code[blockIdx.x - 1];
	int numero_cicli = (els_per_sezione + blockDim.x - 1)/blockDim.x;
	int elemento_limite = min(els_per_sezione*(blockIdx.x+1), nels);

	for (int ciclo = 0; ciclo < numero_cicli;++ciclo, idx += blockDim.x) {
		if (idx < elemento_limite) 
		{
			int4 val = scan[idx];
			int4 val_in=in[idx];
			if(idx==nels-1)
				*max=val.w+correzione_dal_blocco_precedente;
			//trasformazione da scan inclusivo ad esclusivo
		

				val.x += correzione_dal_blocco_precedente - (((val_in.x)&shift)?0:1);
				val.y += correzione_dal_blocco_precedente - (((val_in.y)&shift)?0:1);
				val.z += correzione_dal_blocco_precedente - (((val_in.z)&shift)?0:1);
				val.w += correzione_dal_blocco_precedente - (((val_in.w)&shift)?0:1);
			
			scan[idx] = val;
		}
	}
}
__global__ void reorder(const int4* restrict scan,const int4* restrict in, int* restrict out,int nels,int* max)
{
		int els_per_sezione = (nels + gridDim.x - 1)/gridDim.x;
		int idx = threadIdx.x + blockIdx.x*els_per_sezione;
		int numero_cicli = (els_per_sezione +blockDim.x -1)/blockDim.x;
		int elemento_limite = min(els_per_sezione*(blockIdx.x+1), nels);
		int max_scan=*max;
		int val_scan_succ;
		for (int ciclo = 0; ciclo < numero_cicli;++ciclo, idx += blockDim.x) 
		{
			if (idx < elemento_limite) 
			{
				int4 val_num=in[idx], val_scan=scan[idx];
				
				//printf("\nels/4=%d ,max scan=%d\n,idx=%d\n",nels,max_scan,idx);

				if(val_scan.x!=val_scan.y)  // confronto 1° elemento con 2° elemento della quartina
					out[val_scan.x]=val_num.x;
				else
					out[max_scan  + idx*4 - val_scan.x]=val_num.x;

				if(val_scan.y!=val_scan.z) // confronto 2° elemento con 3° elemento della quartina
					out[val_scan.y]=val_num.y;
				else
					out[max_scan + idx*4+1 - val_scan.y]=val_num.y;

				if(val_scan.z!=val_scan.w) // confronto 3° elemento con 4° elemento della quartina
					out[val_scan.z]=val_num.z;
				else
					out[max_scan  + idx*4+2 - val_scan.z]=val_num.z;

				if(idx!=nels-1)
				{
					val_scan_succ=scan[idx+1].x; // primo valore della quartina successiva

					if(val_scan.w!=val_scan_succ)
						out[val_scan.w]=val_num.w;
					else
						out[max_scan  + idx*4+3 - val_scan.w]=val_num.w;
				}
				else
				{
					if(val_scan.w!=*max)
						out[val_scan.w]=val_num.w;
					else
						out[max_scan + idx*4+3 - val_scan.w]=val_num.w;
				}
			}
		}



}


int main(int argc, char *argv[])
{
	if (argc < 5)
		error("sintassi: RadixSort numels thread_per_blocco numero_blocchi_scan massimo_valore_possibile");

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
	int numMax= atoi(argv[4]); /*  massimo valore possibile*/
	if (numMax <= 0)
		error("il valore massimo possibile deve essere positivo");
	const size_t memsize = sizeof(int)*nels;

	//inizializzazione dei vettori
	int4 *d_v1, *d_scan, *d_code,*d_out,*tmp;
	int numbit;
	int *d_max;
	int flag_primo_lancio=1;
	int shift;
	//calolo fatto avendo il possibile  massimo
	int cicli=log(numMax)/log(2)+1;
	printf("numero cicli da fare=%d\n",cicli);
	//allocazione su GPU dei vettori
	cudaError_t err = cudaMalloc(&d_v1, memsize);
	cuda_check(err, "malloc v1");
	err= cudaMalloc(&d_max,sizeof(int));
	cuda_check(err,"malloc max");
	err = cudaMalloc(&d_scan, memsize);
	cuda_check(err, "malloc scan");
	err = cudaMalloc(&d_code, numBlocksScan*sizeof(int));
	cuda_check(err, "malloc code");
	err = cudaMalloc(&d_out, memsize);
	cuda_check(err, "malloc out");

	//allocazione su CPU dei vettori
	int *vout = (int*)malloc(memsize);
	if (!vout)
		error("alloc vscan");
	//init di numeri random su CPU
	init_random(vout,nels,numMax);
	err = cudaMemcpy(d_v1,vout,memsize, cudaMemcpyHostToDevice);
	cuda_check(err, "memcpy vett su GPU");
	/*
	//inizializzazione su GPU di numeri decrescenti a partire da nels
	int numBlocks = (nels + numThreads - 1)/numThreads;
	init<<<numBlocks, numThreads>>>((int*)d_v1, nels);
	*/
	

	cudaEvent_t before_scan, after_scan;

	//creazione eventi
	err = cudaEventCreate(&before_scan);
	cuda_check(err, "create event before");
	err = cudaEventCreate(&after_scan);
	cuda_check(err, "create event after");
	cudaEventRecord(before_scan);
	//prova ad ottimizzare la cache
	cudaFuncSetCacheConfig(scan_step1,  cudaFuncCachePreferL1);	
	cudaFuncSetCacheConfig(fixup,  cudaFuncCachePreferL1);	
	cudaFuncSetCacheConfig(reorder,  cudaFuncCachePreferL1);

	for(numbit=0;numbit<cicli;numbit++)
	{

		flag_primo_lancio=1;
		shift=1<<numbit;
		err= cudaMemset (d_code,0, numBlocksScan*sizeof(int));
		cuda_check(err, "memset");

		scan_step1<<<numBlocksScan, numThreads, numThreads*sizeof(int)>>>(d_scan, d_v1, nels/4, (int*)d_code,numbit,flag_primo_lancio,shift);

		flag_primo_lancio=0;

		scan_step1<<<1, numThreads, numThreads*sizeof(int)>>>(d_code, d_code, numBlocksScan/4, NULL,0,flag_primo_lancio,0);

		fixup<<<numBlocksScan, numThreads>>>(d_scan, nels/4, (int*)d_code,d_v1,numbit,d_max,shift);

		reorder<<<numBlocksScan,numThreads>>>(d_scan,d_v1,(int*)d_out,nels/4,d_max);
		
		//scambio dei puntatori d_out e d_v1
		if(numbit !=cicli-1)
		{
			tmp=d_v1;
			d_v1=d_out;	
			d_out=tmp;
		}
	}
	cudaEventRecord(after_scan);
	err = cudaEventSynchronize(after_scan);
	cuda_check(err, "after scan sync");
	float runtime_ms;
	cudaEventElapsedTime(&runtime_ms, before_scan, after_scan);
	printf("scan runtime: %.4g ms\n", runtime_ms);
	err = cudaMemcpy(vout, d_out, memsize, cudaMemcpyDeviceToHost);
	cuda_check(err, "memcpy");;
	verify_random(vout,nels)?printf("Ordinamento riuscito!\n"):printf("Ordinamento non riuscito!\n");
	if(nels <=32)
	{
		printf("array ordinato:");
		for(int i=0;i<nels;++i)
			printf("%d ",vout[i]);
	}
	printf("\n");
	
}
