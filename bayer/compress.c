// LossyCompression.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#define  REMOVE_BIT_COUNT(bitcount, bytecount, bits)  { bitcount-=bits;	\
                                                        if(bitcount<0)	\
							{   \
							    bytecount--;    \
							    bitcount+=8;    \
							}   \
							} 
#define  ADD_BIT_COUNT(bitcount, bytecount, bits)  { bitcount+=bits;	\
                                                        if(bitcount>=8)	\
							{   \
							    bytecount++;    \
							    bitcount-=8;    \
							}   \
							} 

unsigned short BYTE_GET_FIELD_HIGHT2LOW(unsigned high,
					unsigned bits,
					unsigned byte,
					unsigned byte_next)
{
    unsigned	char	temp=0, temp1=0;
    if(bits>high+1)
    {
	temp=BYTE_GET_FIELD_HIGHT2LOW(high,high+1,byte, 0);
	temp=temp<<(bits-high-1);
	temp=temp+BYTE_GET_FIELD_HIGHT2LOW(7,bits-high-1,byte_next,0);
	return temp;
    }else
    {
	temp=(byte>>(high-bits+1))&((1<<bits)-1);
	return temp;
    }
}

unsigned short BYTE_GET_FIELD_LOW2HIGH(unsigned low,
					unsigned bits,
					unsigned byte,
					unsigned byte_next)
{
    unsigned	char	temp=0, temp1=0;
    if(bits>8-low)
    {
	temp=BYTE_GET_FIELD_LOW2HIGH(0,bits-8+low,byte_next,0);
	temp=temp<<(8-low);
	temp=temp+BYTE_GET_FIELD_LOW2HIGH(low,8-low,byte, 0);
	return temp;
    }else
    {
	temp=(byte>>(low))&((1<<bits)-1);
	return temp;
    }
}

void decomp_1(unsigned char *buf, unsigned short *buf_out)
{
	int   count;
	int   bitcount=7;
	int   bytecount=26;
	int   exp;
	int   output_count=0;
	count=4;
        for(;count;count--)
	{
	    unsigned short temp=0;
	    exp=BYTE_GET_FIELD_HIGHT2LOW(bitcount,3,buf[bytecount], buf[bytecount-1])+1;
	    REMOVE_BIT_COUNT(bitcount,bytecount,3);
	    temp= BYTE_GET_FIELD_HIGHT2LOW(bitcount,6,buf[bytecount], buf[bytecount-1]);
	    REMOVE_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8]=temp<<exp;
	    temp= BYTE_GET_FIELD_HIGHT2LOW(bitcount,6,buf[bytecount], buf[bytecount-1]);
	    REMOVE_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+2]=temp<<exp;
	    temp= BYTE_GET_FIELD_HIGHT2LOW(bitcount,6,buf[bytecount], buf[bytecount-1]);
	    REMOVE_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+4]=temp<<exp;
	    temp= BYTE_GET_FIELD_HIGHT2LOW(bitcount,6,buf[bytecount], buf[bytecount-1]);
	    REMOVE_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+6]=temp<<exp;


	    exp=BYTE_GET_FIELD_HIGHT2LOW(bitcount,3,buf[bytecount], buf[bytecount-1])+1;
	    REMOVE_BIT_COUNT(bitcount,bytecount,3);
	    temp= BYTE_GET_FIELD_HIGHT2LOW(bitcount,6,buf[bytecount], buf[bytecount-1]);
	    REMOVE_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+1]=temp<<exp;
	    temp= BYTE_GET_FIELD_HIGHT2LOW(bitcount,6,buf[bytecount], buf[bytecount-1]);
	    REMOVE_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+3]=temp<<exp;
	    temp= BYTE_GET_FIELD_HIGHT2LOW(bitcount,6,buf[bytecount], buf[bytecount-1]);
	    REMOVE_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+5]=temp<<exp;
	    temp= BYTE_GET_FIELD_HIGHT2LOW(bitcount,6,buf[bytecount], buf[bytecount-1]);
	    REMOVE_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+7]=temp<<exp;
	    output_count++;
	}
}

#define	MAKE_VLAUE(bits, exp)	((0==exp) ? (bits<<exp): ((bits<<exp)+(1<<(exp-1))))
void decomp_2(unsigned char *buf, unsigned short *buf_out)
{
	int   count;
	int   bitcount=0;
	int   bytecount=0;
	int   exp;
	int   output_count=0;
	count=4;
        for(;count;count--)
	{
	    unsigned short temp=0;
	    exp=BYTE_GET_FIELD_LOW2HIGH(bitcount,3,buf[bytecount], buf[bytecount+1])+1;
	    ADD_BIT_COUNT(bitcount,bytecount,3);

	    temp= BYTE_GET_FIELD_LOW2HIGH(bitcount,6,buf[bytecount], buf[bytecount+1]);
	    ADD_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8]=MAKE_VLAUE(temp,exp);
	    temp= BYTE_GET_FIELD_LOW2HIGH(bitcount,6,buf[bytecount], buf[bytecount+1]);
	    ADD_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+2]=MAKE_VLAUE(temp, exp);
	    temp= BYTE_GET_FIELD_LOW2HIGH(bitcount,6,buf[bytecount], buf[bytecount+1]);
	    ADD_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+4]=MAKE_VLAUE(temp, exp);
	    temp= BYTE_GET_FIELD_LOW2HIGH(bitcount,6,buf[bytecount], buf[bytecount+1]);
	    ADD_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+6]=MAKE_VLAUE(temp, exp);


	    exp=BYTE_GET_FIELD_LOW2HIGH(bitcount,3,buf[bytecount], buf[bytecount+1])+1;
	    ADD_BIT_COUNT(bitcount,bytecount,3);
	    temp= BYTE_GET_FIELD_LOW2HIGH(bitcount,6,buf[bytecount], buf[bytecount+1]);
	    ADD_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+1]=MAKE_VLAUE(temp, exp);
	    temp= BYTE_GET_FIELD_LOW2HIGH(bitcount,6,buf[bytecount], buf[bytecount+1]);
	    ADD_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+3]=MAKE_VLAUE(temp, exp);
	    temp= BYTE_GET_FIELD_LOW2HIGH(bitcount,6,buf[bytecount], buf[bytecount+1]);
	    ADD_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+5]=MAKE_VLAUE(temp, exp);
	    temp= BYTE_GET_FIELD_LOW2HIGH(bitcount,6,buf[bytecount], buf[bytecount+1]);
	    ADD_BIT_COUNT(bitcount,bytecount,6);
	    buf_out[output_count*8+7]=MAKE_VLAUE(temp, exp);
	    output_count++;
	}
}



int main(int argc, char* argv[])
{
	FILE *fp = fopen(argv[1], "rb");
	FILE *outfp = fopen(argv[2], "wb");
	FILE *fptemp= fopen("./test.raw","wb");
	unsigned char  buf[27], buftemp[27];
	unsigned short buf_out[32];
	int   count;
	int   pixel=0;
/*	for(;1;)
	{
	    if(4!=fread(buf,1,4,fp))
		break;
	    buftemp[3]=buf[0];
	    buftemp[2]=buf[1];
	    buftemp[1]=buf[2];
	    buftemp[0]=buf[3];
	    fwrite(buftemp,1,4,fptemp);
	}
	fclose(fptemp);
	fp= fopen("./test.raw","rb");
*/
	for(;1;)
	{
    	count=fread(buf,1,27,fp);
	    if(27!=count)
		break;
	    decomp_1(buf,buf_out);
	    pixel=pixel+32;
	    if(1920+32==pixel)
	    { 
		//break;
	    }
	    count=fwrite(buf_out,1,64,outfp);
	}
	fclose(outfp);
}
