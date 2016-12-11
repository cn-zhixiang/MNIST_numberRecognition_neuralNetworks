## Copyright (C) 2016 王志翔
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} transformDataToImage (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: 王志翔 <wangzhixiang@wangzhixiangdeMacBook-Air.local>
## Created: 2016-12-11

function [] = transformDataToImage ( flag)

[X y] = loadData(flag);
start_time = time;
[m n] = size(X);
number_begin = 1;
number_end = m;

%确定输出目录
if flag ~= 0
  dir = 'MINST/train';
else
  dir = 'MINST/test';
end

%为0~9每个数字的当前索引数组
index = [0:10000:90000]';

for i=number_begin:number_end
  y_i = y(i);
  index_y_i = index(y_i+1);
  index(y_i+1) = index_y_i+1;
  filename = sprintf('%s/%05d.png',dir ,index_y_i);
  image = uint8(reshape(X(i,:), [28 28])'*255);
  imwrite(image, filename);
end

diff = time-start_time;
printf('Writing %d images took %.2f seconds...\n', (number_end-number_begin+1), diff);
endfunction
