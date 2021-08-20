#include "conv_fp.hpp"



//#endif
data_type test_fm_data[8][8][8];
data_type test_w_data[3][3][128][8];
data_type bias[128];
data_type test_output_fm[8][8][128];
data_type pysical_output_fm[8][8][128];
data_type output_abandon[4];
int main()
{

unsigned int PF_test = PASS; // test pass / fail test variable
unsigned int sample_cnt;
unsigned int sample_cnt_test, sample_out,sample_out_1,sample_out_2,sample_out_3;
unsigned int block_cnt;



stream<AXI_VAL_IN> str_in_0,str_in_1,str_in_2,str_in_3;
stream<AXI_VAL_OUT> str_out_0,str_out_1,str_out_2,str_out_3; // stream in, stream out
AXI_VAL_IN fm, fm_1,w,w_1;


ap_uint<16> R=8;  //row of input feature map
ap_uint<16> C=8;  //col of input feature map
ap_uint<16> N=8;   //channel of weight:N=1 when depth_wise=true
ap_uint<16> M=128;  //channel of output feature map

ap_uint<16> Tr=4;  //block size of row of output feature map
ap_uint<16> Tc=4;  //block size of column of output feature map
ap_uint<16> Tm=64;   //block channel of output feature map
ap_uint<16> Tn=4;   //block channel of weight
ap_uint<16> Tri=6; //block size of row of input feature map
ap_uint<16> Tci=6; //block size of column of input feature map

ap_uint<16> Tr_num=2; // block number of input/output feature map in row dimension: ceil[R/Tr]
ap_uint<16> Tc_num=2; // block number of input/output feature map in column dimension: ceil[C/Tc]
ap_uint<16> Tm_num=2; // block number of output feature map in channel dimension: ceil[M/Tm]
ap_uint<16> Tn_num=2; // block number of weight in channel dimension: ceil[N/Tn]

ap_uint<16> K=3;  // kernel sizes supported: 3 or 1
ap_uint<16> S=1;  // only stride 1 or 2 supported
ap_uint<16> P=2;

bool leakyrelu = 1;

ap_uint<16> total;
ap_uint<16> total_1;
ap_uint<16> total_2;
ap_uint<16> total_row;
ap_uint<16> total_col;
ap_uint<16> total_ti;
ap_uint<16> total_ti_1;

network_t *net_CPU;
layer_t layer_test = new layer_t("c1   ",   8, 8,  8, 128,   4, 4, 64, 4,   6, 6 ,   2,  2,  2,  2,  3, 1, 2, 1);
addLayer(net_CPU, layer_test);


// initialize data to write to FIFO=
///////////////////////////
for(ap_uint<16> row=0;row<S*R;row++){
	for(ap_uint<16> col=0;col<S*C;col++){
			for(ap_uint<16> ti=0;ti<N;ti++){
				test_fm_data[row][col][ti]=	(row*C*N+col*N+ti)/10;
//				test_fm_data[row][col][ti]=	1;
//				cout<<"test_fm_data "<< test_fm_data[row][col][ti]<<endl;
			}}}
for(ap_uint<16> i=0;i<K;i++){
	for(ap_uint<16> j=0;j<K;j++){
		for(ap_uint<16> to=0;to<M;to++){
			for(ap_uint<16> ti=0;ti<N;ti++){
				  test_w_data[i][j][to][ti]=(i*K*M*N+j*M*N+to*N+ti)/10;
//				  test_w_data[i][j][to][ti]=1;
//				cout<<"test_w_data is "<< test_w_data[i][j][to][ti]<<endl;

			}}}}
for(ap_uint<16> to=0;to<M;to++){
	bias[to]=1;
}
/////////////computation the correct output//////////////////////
		ROW_LOOP:for(ap_uint<16> row=0;row<S*R;row+=S){
		COL_LOOP:	for(ap_uint<16> col=0;col<S*C;col+=S){
		CH_OUT_LOOP:	for(ap_uint<16> to=0;to<M;to++){
		CH_IN_LOOP:			for(ap_uint<16> ti=0;ti<N;ti++){
								for(ap_uint<16> i=0; i<K;i++){
									for(ap_uint<16> j=0;j<K;j++){
										ap_uint<16> row_total;
										ap_uint<16> col_total;
										row_total= (K==1)? ap_uint<16>(row+i):ap_uint<16>(row+i-1);
										col_total= (K==1)? ap_uint<16>(col+j):ap_uint<16>(col+j-1);
			if((ti==0)&&(i==0)&&(j==0))
				test_output_fm[row/S][col/S][to]=bias[to];
			if((row_total<0)||(col_total<0)||(row_total>=S*R)||(col_total>=S*C))
				test_output_fm[row/S][col/S][to]+=0;
			else
				test_output_fm[row/S][col/S][to]+=test_w_data[i][j][to][ti]*test_fm_data[row_total][col_total][ti];

		}}}
//		   cout<<"test_output_fm   "<<"["<<row<<"]"<<"["<<col<<"]"<<"["<<to<<"]"<<"is "<< test_output_fm[row][col][to]<<endl;
		}}}

// run test loop
///////////////////////////write_data////////////////////////////////////////////////////////////////

write_data_ROW_LOOP:for(ap_uint<16> row=0;row<S*R;row+=S*Tr){
	write_data_COL_LOOP:	for(ap_uint<16> col=0;col<S*C;col+=S*Tc){
		write_data_CH_OUT_LOOP:	for(ap_uint<16> to=0;to<M;to+=Tm){
			write_data_CH_IN_LOOP:		for(ap_uint<16> ti=0;ti<N;ti+=Tn){

			   write_pixcel:			     for(ap_uint<16> trr=0;trr<S*Tri;trr++){
												for(ap_uint<16> tcc=0;tcc<S*Tci;tcc++){
													for(ap_uint<16> tii=0;tii<Tn;tii+=2){
														ap_uint<16> ti_total=(ti+tii);
														ap_uint<16> ti_total_1=(ti+tii+1);

														ap_uint<16> row_total;
														ap_uint<16> col_total;
														row_total= (K==1)? ap_uint<16>(row+trr):ap_uint<16>(row+trr-1);
														col_total= (K==1)? ap_uint<16>(col+tcc):ap_uint<16>(col+tcc-1);

															if((ti_total>=N)||((row_total)<0)||((col_total)<0)||(row_total>=S*R)||(col_total>=S*C)){
																  fm.data=0;
																  fm_1.data=0;
																  str_in_0.write(fm) ;
																  str_in_1 << fm_1;
//																  cout<<"fm "<< fm.data<<endl;
//																  cout<<"fm_1 "<< fm_1.data<<endl;
															}
															else if(ti_total_1==N){
																  fm.data=test_fm_data[row_total][col_total][ti_total];
																  fm_1.data=0;
																  str_in_0 << fm;
																  str_in_1 << fm_1;
//																  cout<<"fm "<< fm.data<<endl;
//																  cout<<"fm_1 "<< fm_1.data<<endl;
															}

															else{
																  fm.data=test_fm_data[row_total][col_total][ti_total];
																  fm_1.data=test_fm_data[row_total][col_total][ti_total_1];
																  str_in_0 << fm;
																  str_in_1 << fm_1;
//																  cout<<"fm "<< fm.data<<endl;
//																  cout<<"fm_1 "<< fm_1.data<<endl;
															}
//

													}}}





            write_weight:				 for(ap_uint<16> i=0;i<K;i++){
											for(ap_uint<16> j=0;j<K;j++){
												for(ap_uint<16> tii=0;tii<(Tn+1);tii++){
													for(ap_uint<16> too=0;too<Tm;too+=2){


															ap_uint<16> ti_total=ti+tii;
															ap_uint<16> to_total=to+too;
															ap_uint<16> to_total_1=to+too+1;

																if(  (to_total>=M) || (   (ti_total>=N)&&(tii < Tn) )  ){
																	w.data=0;
																	w_1.data=0;
																	str_in_2 << w;
																	str_in_3 << w_1;
//																	  cout<<"fm "<< w.data<<endl;
//																	  cout<<"fm_1 "<< w_1.data<<endl;
																}


																else if((to_total_1==M)){
																	w.data=test_w_data[i][j][to_total][ti_total];
																	w_1.data=0;
																	str_in_2 << w;
																	str_in_3 << w_1;
//																	  cout<<"fm "<< w.data<<endl;
//																	  cout<<"fm_1 "<< w_1.data<<endl;
																}

																else if(tii == Tn){
																	w.data=bias[to_total];
																	w_1.data=bias[to_total_1];
																	str_in_2 << w;
																	str_in_3 << w_1;
																}

																else {
																	w.data=test_w_data[i][j][to_total][ti_total];
																	w_1.data=test_w_data[i][j][to_total_1][ti_total];
																	str_in_2 << w;
																	str_in_3 << w_1;
//																	 cout<<"fm "<< w.data<<endl;
//																	 cout<<"fm_1 "<< w_1.data<<endl;
																}
//															}
													}}}}


}}}}





		convolution_fp(
								  layer_test,
								  str_in_0,
								  str_in_1,
								  str_in_2,
								  str_in_3,

								  str_out_0,
								  str_out_1,
								  str_out_2,
								  str_out_3
						  );

		for(ap_uint<16> row=0;row<R;row+=Tr){
			for(ap_uint<16> col=0;col<C;col+=Tc){
				for(ap_uint<16> to=0;to<M;to+=Tm){
						for(ap_uint<16> trr=0;trr<Tr;trr++){
							for(ap_uint<16> tcc=0;tcc<Tc;tcc++){
								for(ap_uint<16> too=0;too<Tm;too+=4){
									ap_uint<16> row_total=row+trr;
									ap_uint<16> col_total=col+tcc;
									ap_uint<16> to_total=to+too;
									ap_uint<16> to_total_1=to+too+1;
									ap_uint<16> to_total_2=to+too+2;
									ap_uint<16> to_total_3=to+too+3;
									AXI_VAL_OUT fmo;
									AXI_VAL_OUT fmo_1;
									AXI_VAL_OUT fmo_2;
									AXI_VAL_OUT fmo_3;
									fmo=str_out_0.read();
									fmo_1=str_out_1.read();
									fmo_2=str_out_2.read();
									fmo_3=str_out_3.read();
									if((row_total>=R)||(col_total>=C)||(to_total>=M)){
										output_abandon[0]=fmo.data;
										output_abandon[1]=fmo_1.data;
										output_abandon[2]=fmo_2.data;
										output_abandon[3]=fmo_3.data;
									}
									else if(to_total_3==M){
										pysical_output_fm[row+trr][col+tcc][to_total]=fmo.data;
										pysical_output_fm[row+trr][col+tcc][to_total_1]=fmo_1.data;
										pysical_output_fm[row+trr][col+tcc][to_total_2]=fmo_2.data;
										output_abandon[3]=fmo_3.data;

									}
									else if(to_total_2==M){
										pysical_output_fm[row+trr][col+tcc][to_total]=fmo.data;
										pysical_output_fm[row+trr][col+tcc][to_total_1]=fmo_1.data;
										output_abandon[2]=fmo_2.data;
										output_abandon[3]=fmo_3.data;
//										if(fmo.last==1){
//											cout<<"pysical_output_fm   "<<"["<<row<<"]"<<"["<<col<<"]"<<"["<<to<<"]"<<"is "<< test_output_fm[row_total][col_total][to_total]<<endl;
//										}

									}
									else if(to_total_1==M){
										pysical_output_fm[row+trr][col+tcc][to_total]=fmo.data;
										output_abandon[1]=fmo_1.data;
										output_abandon[2]=fmo_2.data;
										output_abandon[3]=fmo_3.data;
//										if(fmo.last==1){
//											cout<<"pysical_output_fm   "<<"["<<row<<"]"<<"["<<col<<"]"<<"["<<to<<"]"<<"is "<< test_output_fm[row_total][col_total][to_total]<<endl;
//										}

									}
									else{
										pysical_output_fm[row+trr][col+tcc][to_total]=fmo.data;
										pysical_output_fm[row+trr][col+tcc][to_total_1]=fmo_1.data;
										pysical_output_fm[row+trr][col+tcc][to_total_2]=fmo_2.data;
										pysical_output_fm[row+trr][col+tcc][to_total_3]=fmo_3.data;
//										if(fmo.last==1){
//											cout<<"pysical_output_fm   "<<"["<<row<<"]"<<"["<<col<<"]"<<"["<<to<<"]"<<"is "<< test_output_fm[row_total][col_total][to_total]<<endl;
//										}

									}
									if(fmo.last==1){
										cout<<"last is 1 "<< "["<<row_total<<"]"<<"["<<col_total<<"]"<<"["<<to_total<<"]"<<endl;
									}
									if(fmo_1.last==1){
										cout<<"last is 1 "<< "["<<row_total<<"]"<<"["<<col_total<<"]"<<"["<<to_total_1<<"]"<<endl;
									}
									if(fmo_2.last==1){
										cout<<"last is 1 "<< "["<<row_total<<"]"<<"["<<col_total<<"]"<<"["<<to_total_2<<"]"<<endl;
									}
									if(fmo_3.last==1){
										cout<<"last is 1 "<< "["<<row_total<<"]"<<"["<<col_total<<"]"<<"["<<to_total_3<<"]"<<endl;
									}
								}}}}}}
//#endif
/////////////////constrast//////////////////////
       for(ap_uint<16> row=0;row<R;row++){
    	   for(ap_uint<16> col=0;col<C;col++){
    		   for(ap_uint<16> to=0;to<M;to++){
    		//	   if((row<3)&&(col<3)&&(to<64)){
   	//		   if((pysical_output_fm[row][col][to] != test_output_fm[row][col][to])){
    				   cout<<"test_output_fm   "<<"["<<row<<"]"<<"["<<col<<"]"<<"["<<to<<"]"<<"is "<< test_output_fm[row][col][to]<<endl;
    				   cout<<"pysical_output_fm"<<"["<<row<<"]"<<"["<<col<<"]"<<"["<<to<<"]"<<"is "<< pysical_output_fm[row][col][to]<<endl;
   	//		   }
    				   PF_test = PASS;
    			  // cout<<"row="<< row<<"col="<<col<<"to="<<to<<endl;

    			 //  printf("output_fm %d%d%dFAILED to gain the correct convolution result \n",row,col,to);

//    			   }
    				   }}}

//	if (axi_interrupt) { // && !axi_cpuRelease) // for C sim only - do not check for !axi_cpuRelease
//		cout << "----------------------  AXI STREAM INTERRUPT SET -------------------------\n";
//		for(sample_cnt_test = 0; sample_cnt_test < BLOCK_SIZE; sample_cnt_test++) { // test to verify FIFO data matches test_data
////#ifdef AXI_MASTER
////			if ( (pingPongBuffer[sample_cnt_test] != test_data[sample_cnt_test+block_cnt*BLOCK_SIZE]) )
////#else
//			sample_out = str_out.read();
//			sample_out_1 = str_out_1.read();
//			sample_out_2 = str_out_2.read();
//			sample_out_3 = str_out_3.read();
//			// pop output value off stream
//			// cout << "0x" << setfill('0')  << setw(6) << hex << sample_out << " ";
//			if ( (sample_out != test_w_data[sample_cnt_test+block_cnt*BLOCK_SIZE]) )
////#endif
//			{
//				PF_test = PASS;
//				cout << "0x" << setfill('0')  << setw(6) << hex << sample_out << " ";
//				cout << "-----------------------------------------------\n";
//				cout << "FAILED to Match AXI Stream Memory Location \n";
//				cout << "-----------------------------------------------\n";
//				break;
//			}
//		}
		if (PF_test == FAIL) {
			cout << endl;
			cout << "-----------------------------------------------\n";
			   cout << "FAILED to gain the correct convolution result \n";
			cout << "-----------------------------------------------\n";

		}
//        cout << endl;
//		axi_cpuRelease = 1;
//	  } // axi_interrupt
//	  else axi_cpuRelease = 0;
//
//} // block_cnt
//
//// print sysmem
//cout << "-----------------------------------------------\n";
//cout << "-----------------------------------------------\n";
//
//if (PF_test == FAIL) {
//	cout << endl;
//	cout << "FIFO TEST FAILED \n";
//}
//else
//	cout << "FIFO TEST PASSED \n";
//	cout << "-----------------------------------------------\n";
//	cout << "-----------------------------------------------\n";
//	cout << endl;

return PF_test;

} // end function main
