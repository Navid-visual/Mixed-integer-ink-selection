%% Load the data
% Mixed Integer Linear Constrained Quadratic Programming
% Load the ink names 
load('.\Test data\ink_names.mat')
% Load the transmittance library 
load('.\Test data\transmitance_completelibrary.mat')
n = 44; % number of all inks
o = 8; %number of active inks
m = n-o; %number of deactive inks

% Load the coreset test data 
coresettrans = struct2cell(load('.\Test data\coresettrans_cat'));
coresettrans = coresettrans{1};
%% Initialize the optimization variables and values
coreset_absorbtance=trans2absorbtance(coresettrans); % Convert the coreset transmittances to absorbance
absorbtance_completelibrary=trans2absorbtance(transmitance_completelibrary); % Convert the coreset transmittances to absorbance

size_of_test=size(coreset_absorbtance,1);

w = sdpvar(size_of_test,n); % Initialize the continues variables of thicknesses.
x = binvar(1,n); % Initialize the binary variable of selection.
Z = sdpvar(size(coreset_absorbtance',1),size(coreset_absorbtance',2)); 

%% Adjust the optimization parameters and define the objective value and the constraints.
constraints = [ 0 <= w <= 4*(repmat(x,[size_of_test 1])),... % Equation (7g)
                sum(x) <= o ,... % Equation (7e)
                Z >= (absorbtance_completelibrary'*w'-coreset_absorbtance'), ... % Equation (7b)
                Z >= -(absorbtance_completelibrary'*w'-coreset_absorbtance'), ... % Equation (7c)
                ];
objective = sum(sum(Z)); % Equation (7a)
options = sdpsettings('solver','gurobi','debug','on');
sol = optimize(constraints,objective,options);


if sol.problem == 0
    selected_inks = double(x); % Return the index of the selected inks.
    thickness = double(w); % Return the calculated thickness of ink layers.
    double(objective) % Return the final objective value.
    names(logical(selected_inks)) % Find the name of the selected inks.
else
    sol.info  % Return the report of the optimization  
end