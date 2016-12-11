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
## @deftypefn {Function File} {@var{retval} =} predict_nn (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: 王志翔 <wangzhixiang@wangzhixiangdeMacBook-Air.local>
## Created: 2016-12-07

function [pred] = predict_nn(theta, n_2, X)

[m n_1] = size(X);
k = 10;

theta_1 = reshape(theta(1:(n_1+1)*n_2), [n_1+1 n_2]);
theta_2 = reshape(theta((n_1+1)*n_2+1:end), [n_2+1 k]);

z_2 = [ones(m,1) X]*theta_1;
a_2 = [ones(m,1) sigmoid(z_2)];
z_3 = a_2*theta_2;
a_3 = sigmoid(z_3);

[temp pred] = max(a_3, [], 2);
pred = pred-1;

endfunction
