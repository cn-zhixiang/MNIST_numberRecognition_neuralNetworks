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
## @deftypefn {Function File} {@var{retval} =} fmin_nn (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: 王志翔 <wangzhixiang@wangzhixiangdeMacBook-Air.local>
## Created: 2016-12-07

function [theta] = fmin_nn (n_2, X, y, lambda)

lr_rate=1;

[m n_1] = size(X);
k=10;
theta_1_init = (rand(n_1+1, n_2)-0.5)/10;
theta_2_init = (rand(n_2+1, k)-0.5)/10;
theta = [theta_1_init(:); theta_2_init(:)];

number_epoch = 5;
size_batch = 1000;
number_batch = m/1000;
fprintf('Mini batch gradient decrease begin...\n')
for iter = 1:number_epoch
  for i = 1 : number_batch
    X_mini = X(size_batch*(i-1)+1:size_batch*i, :);
    y_mini = y(size_batch*(i-1)+1:size_batch*i, :);
    [J grad] = lrCostFunction_nn(theta, n_2, X_mini, y_mini, lambda);
    fprintf(1,'J now is %f\n',J);
    theta = theta - lr_rate*grad;
   end
end

fprintf('\nFull batch gradient decrease begin...\n')
for i = 1:20
  [J grad] = lrCostFunction_nn(theta, n_2, X, y, lambda);
  fprintf(1,'J now is %f\n',J);
  theta = theta - lr_rate/10*grad;
end

endfunction
