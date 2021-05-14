#!/bin/bash
# wget http://www.summarization.com/mead/download/MEAD-3.12.tar.gz
tar xzf MEAD-3.12.tar.gz

mkdir dependencies
cd dependencies

sudo apt-get update
sudo apt-get install -y build-essential

# Expat (install this before XML::Parser)
wget https://sourceforge.net/projects/expat/files/expat/2.2.9/expat-2.2.9.tar.gz
tar xzf expat-2.2.9.tar.gz
cd expat-2.2.9
./configure
make
sudo make install
# 
# # XML::Parser - required
# wget http://www.cpan.org/authors/id/C/CO/COOPERCL/XML-Parser.2.30.tar.gz
# tar xzf XML-Parser.2.30.tar.gz
# 

cd ../..


# Comprehensive Perl Archive Network (CPAN)
# https://www.cpan.org/
#
# execute `cpan` for first time configuration
#
# https://stackoverflow.com/questions/22360091/how-to-fix-yaml-not-installed-when-installing-xmlsimple
#
# $ sudo cpan
# go into interactive prompt
# > install YAML
# > install CPAN
# > reload cpan
#
# 
# > install XML::Parser
# > install XML::Writer
# > install XML::TreeBuilder
# > install Text::Iconv

# or get cpanm first (recommend)
# > install App::cpanminus

sudo cpanm XML::Parser XML::Writer XML::TreeBuilder Text::Iconv

cd mead
sudo perl Install.PL
