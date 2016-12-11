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
## @deftypefn {Function File} {@var{retval} =} lrCostFunction_nn (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: 王志翔 <wangzhixiang@wangzhixiangdeMacBook-Air.local>
## Created: 2016-12-06

function [J grad] = lrCostFunction_nn(theta_row, n_2, X, y, lambda)

%X 大小为 m * n, theta_1 大小为 n_1+1 * n_2, theta_2 大小为 n_2+1 * k
[m n_1] = size(X);

k=10;
theta_1 = reshape(theta_row(1 : (n_1+1)*n_2), [n_1+1, n_2]);
theta_2 = reshape(theta_row((n_1+1)*n_2+1:end), [n_2+1, k]);

%前向传导, X 大小为 m * n_1+1, z_2 大小为 m * n_2, a_2 大小为 m * n_2+1
z_2 = [ones(m,1) X]*theta_1;
a_2 = [ones(m,1) sigmoid(z_2)];
h = sigmoid(a_2*theta_2);
l = (repmat([0:9], m, 1) == repmat(y,1,k));

%计算代价函数
J = -1/m*sum(sum(l.*log(h)+(1-l).*log(1-h)));
J = J+ lambda/(2*m)*(sum(sum(theta_1(2:end,:).^2))+sum(sum(theta_2(2:end,:).^2)));

%初始化 grad
grad_1 = zeros(n_1+1, n_2);
grad_2 = zeros(n_2+1, k);

%反向传导, delta_3 大小为 m * k, delta_2 大小为 m * n_2
delta_3 = (h-l);
delta_2 = (delta_3 * theta_2')(:,2:end) .* sigmoidGradient(z_2);
grad_2 = 1/m* a_2' * delta_3;
grad_1 = 1/m* [ones(m,1) X]' * delta_2;

%正则化
grad_2(2:end,:) = grad_2(2:end,:)+lambda/m*theta_2(2:end,:);
grad_1(2:end,:) = grad_1(2:end,:)+lambda/m*theta_1(2:end,:);

grad = [grad_1(:);grad_2(:)];

endfunction
