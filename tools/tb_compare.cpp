// Compare Pyrrhic WDL against python-chess reference lines "<fen> <wdl -2..2>" on stdin (see scripts/gen_tb_positions.py).
#include "tablebase.h"
#include <cstdio>
#include <iostream>
#include <sstream>
int main(){ Bitboards::init(); Zobrist::init(); TB::init(std::string(getenv("HOME"))+"/data/syzygy/tb345");
 std::string line; int bad=0,n=0; while(std::getline(std::cin,line)){ size_t sp=line.rfind(' '); int ref=std::stoi(line.substr(sp+1)); Position p; p.set_fen(line.substr(0,sp));
  int w=TB::probe_wdl(p); int mine = w<0?-99:w-2; ++n; if(mine!=ref){ if(bad++<8) printf("MISMATCH %s ref=%d mine=%d\n",line.substr(0,sp).c_str(),ref,mine);} }
 printf("%d/%d mismatches\n",bad,n); }
