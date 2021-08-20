#include "conv_fp.hpp"
//记得上传到github
void load_input_feature(
		stream<AXI_VAL_IN> &str_in_0,
		stream<AXI_VAL_IN> &str_in_1,
		dimension_t layer_tri,
		dimension_t layer_tci,
		dimension_t layer_tn)
{
	for(dimension_t tri=0;tri<layer_tri;tri++){
		for(dimension_t tci=0;tci<layer_tci;tci++){
			for(dimension_t tn=0;tn<layer_tn;tn+=2){
				dimension_t tn_tmp = tn;
				dimension_t tn_tmp1 = tn + 1;

				IBRAM[tri][tci][tn_tmp] = str_in_0.read().data;
				IBRAM[tri][tci][tn_tmp1] = str_in_1.read().data;
			}
		}
	}

}
void load_weights(
		stream<AXI_VAL_IN>&str_in_2,
		stream<AXI_VAL_IN>&str_in_3,
		ap_uint<3> layer_ki,
		ap_uint<3> layer_kj,
		dimension_t layer_tn,
		dimension_t layer_tm)
{
	for(ap_uint<3> i = 0; i<layer_ki; i++){
		for(ap_uint<3> j=0; j<layer_kj; j++){
			for(dimension_t tn = 0; tn<layer_tn; tn++){
				for(dimension_t tm = 0; tm<layer_tm; tm+=2){
					dimension_t tm_tmp = tm;
					dimension_t tm_tmp1 = tm + 1;

					WBRAM[i][j][tm_tmp][tn] = str_in_2.read().data;
					WBRAM[i][j][tm_tmp1][tn] = str_in_3.read().data;
				}
			}
		}
	}
}

void load_bias(
		stream<AXI_VAL_IN>&str_in_2,
		stream<AXI_VAL_IN>&str_in_3,
		dimension_t layer_tr,
		dimension_t layer_tc,
		dimension_t layer_tm)
{
	for(dimension_t tm = 0;tm<layer_tm;tm+=2){
		dimension_t tm_tmp = tm;
		dimension_t tm_tmp1 = tm + 1;

		data_type tm_tmp_data = str_in_2.read().data();
		data_type tm_tmp_data1 = str_in_3.read().data();
		for(dimension_t tr = 0; tr<layer_tr;tr++){
			for(dimension_t tc = 0;tc<layer_tc;tc++){
				OBRAM[tr][tc][tm_tmp] = tm_tmp_data;
				OBRAM[tr][tc][tm_tmp1] = tm_tmp_data;
			}
		}
	}
}
void macc(
		ap_uint<3> layer_ki,
		ap_uint<3> layer_kj,
		dimension_t layer_tr,
		dimension_t layer_tc,
		dimension_t layer_tm,
		dimension_t layer_tn,
		ap_uint<2> layer_s)
{
	for(ap_uint<3> i=0;i<layer_ki;i++){
		for(ap_uint<3> j=0;j<layer_kj;j++){
			for(dimension_t tr = 0; tr<layer_tr;tr++){
				for(dimension_t tc = 0;tc<layer_tc;tc++){
					for(dimension_t tm = 0;tm < layer_tm;tm++){
						for(dimension_t tn = 0;tn<layer_tn;tn++){
							dimension_t x_center = tr*layer_s + i;
							dimension_t y_center = tc*layer_s + j;
							OBRAM[tr][tc][tm] += IBRAM [x_center][y_center][tn] * WBRAM [i][j][tm][tn];
						}
					}
				}
			}
		}
	}
}
void output(
		stream<AXI_VAL_OUT> str_out_0,
		stream<AXI_VAL_OUT> str_out_1,
		stream<AXI_VAL_OUT> str_out_2,
		stream<AXI_VAL_OUT> str_out_3,
		dimension_t layer_tr,
		dimension_t layer_tc,
		dimension_t layer_tm,
		bool relu,
		bool last_block)
{
	for(dimension_t tr = 0; tr < layer_tr; tr++){
		for(dimension_t tc =0 ; tc < layer_tc; tc++){
			for(dimension_t tm = 0; tm < layer_tm; tm+=4){
				dimension_t tm_tmp = tm;
				dimension_t tm_tmp1 = tm+1;
				dimension_t tm_tmp2 = tm+2;
				dimension_t tm_tmp3 = tm+3;

				AXI_VAL_OUT out_0;
				AXI_VAL_OUT out_1;
				AXI_VAL_OUT out_2;
				AXI_VAL_OUT out_3;

				if( (relu)&&(OBRAM[tr][tc][tm_tmp]<0)){
					out_0.data =  0;
				}
				else{
					out_0.data  =  OBRAM[tr][tc][tm_tmp];
				}

				if( (relu)&&(OBRAM[tr][tc][tm_tmp1]<0)){
					out_1.data =  0;
				}
				else{
					out_1.data  =  OBRAM[tr][tc][tm_tmp1];
				}
				if( (relu)&&(OBRAM[tr][tc][tm_tmp2]<0)){
					out_2.data =  0;
				}
				else{
					out_2.data  =  OBRAM[tr][tc][tm_tmp2];
				}
				if( (relu)&&(OBRAM[tr][tc][tm_tmp3]<0)){
					out_3.data =  0;
				}
				else{
					out_3.data  =  OBRAM[tr][tc][tm_tmp3];
				}

				if(last_block&&(tm==(layer_tm-4))&&(tr==(layer_tr-1))&&(tc==( layer_tc-1))){
                    out_1.last=1;
                    out_1.last=1;
                    out_2.last=1;
                    out_3.last=1;
				}
				else{
                    out_1.last=0;
                    out_1.last=0;
                    out_2.last=0;
                    out_3.last=0;
				}
				str_out_0.write(out_0);
				str_out_1.write(out_1);
				str_out_2.write(out_2);
				str_out_3.write(out_3);
			}
		}
	}
}

void convolution_fp(
		layer_t layer,
		stream<AXI_VAL_IN> &str_in_0,
		stream<AXI_VAL_IN> &str_in_1,
		stream<AXI_VAL_IN> &str_in_2,
		stream<AXI_VAL_IN> &str_in_3,

		stream<AXI_VAL_OUT> &str_out_0,
		stream<AXI_VAL_OUT> &str_out_1,
		stream<AXI_VAL_OUT> &str_out_2,
		stream<AXI_VAL_OUT> &str_out_3,
		)
{
	R_LOOP:for(number_t r=0;r<layer.Tr_num;r++){
		C_LOOP:for(number_t c=0;c<layer.Tc_num;c++){
			M_LOOP:for(number_t m=0;m<layer.Tm_num;m++){
				N_LOOP:for(number_t n=0;n<layer.Tn_num;n++){
					if(n == 0)load_bias(str_in_2, str_in_3, layer.Tr, layer.Tc ,layer.Tm);
					load_input_feature(str_in_0, str_in_1, layer.Tri, layer.Tci, layer.Tn);
					load_weights(str_in_2, str_in_3, layer.kernel, layer.kernel, layer.Tm, layer.Tn);
					macc(layer.kernel,layer.kernel,layer.Tr,layer.Tc,layer.Tn,layer.Tm,layer.stride);
				}
				bool last_block = ((r==(layer.Tr_num-1))&&(c==(layer.Tc_num-1))&&(m==(layer.Tm_num-1)))? true:false;
				output(str_out_0,str_out_1,str_out_2,str_out_3, layer.Tr, layer.Tc, layer.Tm, layer.relu,last_block );
			}
		}
	}
}
