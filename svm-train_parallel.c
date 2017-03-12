#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "svm.h"
#include "mpi.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void exit_with_help()
{
	printf("Usage: svm-train training_set_file model_file\n");
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void set_param(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_node *x_space;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}
void set_Cost(double C[16])
{
	C[0] = 0.03125;
	for (int i =1; i < 16; i++)
		C[i] = C[i-1]*2;
}
void safe_exit()
{
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);

}
void set_Gamma(double G[16])
{
	G[15] = 8.0;
	for (int i = 14; i >= 0; i--)
	{
		G[i] = G[i+1]/2.0;
	} 
}


double** alloc_Accuracy_matrix(int p)
{
	double** matrix;
    	matrix = (double**) malloc(p * sizeof(double *));
    	matrix[0] = (double*) malloc(p * 16 * sizeof(double));
    	for (int i = 1; i < p; i++)
        	matrix[i] = matrix[0] + i*16;
    
	return matrix;
}

void initialize_Accuracy_matrix(double**matrix, int p)
{
	for (int i = 0; i < p; i++)
        	for (int j = 0; j < 16; j++) {
            		matrix[i][j] = 0.0;
        }
	
}

double cross_validate()
{
	int i;
	int total_correct = 0;
	double *target = Malloc(double,prob.l);

	svm_cross_validation(&prob,&param,2,target);

	for(i=0;i<prob.l;i++)
		if(target[i] == prob.y[i])
			++total_correct;

	free(target);
	return 100.0*total_correct/prob.l;
}

void find_global_best(double** Global_Accuracy,int* bestC,int* bestG)
{
	double Max = 0;

	for (int i = 0; i < 16; i++)
		for ( int j = 0; j < 16; j++)
		{
			if(Max < Global_Accuracy[i][j])
			{
				Max = Global_Accuracy[i][j];
				*bestC = i;
				*bestG = j;
			}
		} 

}

void save_matrix(double** matrix)
{
	FILE *fp = fopen("AccuracyGrid","w");
	if (fp == NULL)
		return;
	for (int i = 0; i < 16; i++)
	{
        	for (int j = 0; j < 16; j++)
            		fprintf(fp,"%g ", matrix[i][j]);
        	fprintf(fp,"\n");
   	}
}

int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;
	double Gamma[16];
	double Cost[16];
	int p, my_rank;
	set_param(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);
	error_msg = svm_check_parameter(&prob,&param);


	MPI_Init(NULL,NULL);
	MPI_Comm_size(MPI_COMM_WORLD,&p);
	if(error_msg)
	{
		fprintf(stderr,"ERROR: %s\n",error_msg);
		safe_exit();
	}
	else
	{

		printf("Till here");
		set_Gamma(Gamma);
		set_Cost(Cost);
		double **Global_Accuracy = alloc_Accuracy_matrix(16);



	
		if ( p > 16)
		{
			printf("Too many number of processes!!");
			safe_exit();
		}

		MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

		int my_start = my_rank*(16/p);
		int my_end = (my_rank+1)*(16/p);
		
		double **Accuracy = alloc_Accuracy_matrix((int)16/p);
		initialize_Accuracy_matrix(Accuracy,(int)16/p);
		
		for ( int i = my_start; i < my_end; i++)
			for ( int j = 0; j < 16; j++)
			{
				//struct svm_model *model;
				param.gamma = Gamma[j];	
				param.C = Cost[i];
				printf("Cost = %f, Gamma = %f",param.C,param.gamma);			
				Accuracy[i-my_start][j] = cross_validate();
			}
		// MPI thread sends accuracy for various pair of cost and gamma value.
		MPI_Gather( Accuracy[0], (16/p)*16, MPI_DOUBLE, Global_Accuracy[0],(16/p)*16, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if ( my_rank == 0)
		{
			int bestC = 0;
			int bestG = 0;

			//Write Accuracy Matrix to file
			save_matrix(Global_Accuracy);
			find_global_best(Global_Accuracy,&bestC,&bestG);

			struct svm_model *model;
			param.gamma = Gamma[bestG];
			param.C = Cost[bestC];

			// train on global best and save the model
			model = svm_train(&prob,&param);

			if(svm_save_model(model_file_name,model))
			{
				fprintf(stderr, "can't save model to file %s\n", model_file_name);
				safe_exit();
			}
			
			svm_free_and_destroy_model(&model);
		}
	}

	MPI_Finalize();
	safe_exit();
	
	return 0;
}

void set_param(int argc, char **argv, char *input_file_name, char *model_file_name)
{

	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	if(argc != 3){
		exit_with_help();
	}	

	svm_set_print_string_function(NULL);

	// determine filenames

	strcpy(input_file_name, argv[1]);
	
	strcpy(model_file_name,argv[2]);
}

/* read in a problem (in svmlight format)

	The following code to read input file was taken from libsvm demo train
*/

void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}
