format long g
A = importdata("./tracking/case_24_points_new.txt");
X = A(:,1);
Y = A(:,2);
fig1 = figure;
plot(X,Y,'.-b');
legend('Ground Truth');
hold on


numSteps = numel(X);
dt = 0.1;


truePos = zeros(3, numel(X));
for i = 1:numSteps
    truePos(:,i) = [X(i);Y(i);0];
end

s = rng;
rng(2022);
positionSelector = [1 0 0 0 0 0;0 0 1 0 0 0;0 0 0 0 1 0]; % Position from state
measNoise = randn(size(truePos));
measPos = truePos + measNoise;  % 3¡Á1457
initialState = positionSelector' * measPos(:,1);
initialCovariance = diag([1,1e4,1,1e4,1,1e4]);   % Velocity is not measured

cvekf1 = trackingEKF(@constvel, @cvmeas, initialState, ...
    'StateTransitionJacobianFcn', @constveljac, ...
    'MeasurementJacobianFcn', @cvmeasjac, ...
    'StateCovariance', initialCovariance, ...
    'HasAdditiveProcessNoise', false, ...
    'ProcessNoise', eye(3));

dist = zeros(1,numSteps);
estPos = zeros(3,numSteps);
for i = 2:size(measPos,2)   % 2:1457
    predict(cvekf1, dt);
    dist(i) = distance(cvekf1,truePos(:,i)); % Distance from true position
    estPos(:,i) = positionSelector * correct(cvekf1, measPos(:,i));
end

figure(fig1);
plot(estPos(1,:),estPos(2,:),'.g','DisplayName','Contant Velocity')
hold on
title('True and Estimated Positions')
axis([12.9490 12.9550 4.8195 4.8242] * 1e6)


fig2 = figure;
hold on
plot((1:numSteps)*dt, dist,'g','DisplayName', 'Contant Velocity')
title('Normalized Distance From Estimated Position to True Position')
xlabel('Time (s)')
ylabel('Normalized Distance')
legend

%%%%%%%%%%%%%Constant velocity with high process noise%%%%%%%%%%%%%%
% cvekf2 = trackingEKF(@constvel, @cvmeas, initialState, ...
%     'StateTransitionJacobianFcn', @constveljac, ...
%     'MeasurementJacobianFcn', @cvmeasjac, ...
%     'StateCovariance', initialCovariance, ...
%     'HasAdditiveProcessNoise', false, ...
%     'ProcessNoise', diag([50,50,1])); % Large uncertainty in the horizontal acceleration
% dist = zeros(1,numSteps);
% estPos = zeros(3,numSteps);
% for i = 2:size(measPos,2)
%     predict(cvekf2, dt);
%     dist(i) = distance(cvekf2,truePos(:,i)); % Distance from true position
%     estPos(:,i) = positionSelector * correct(cvekf2, measPos(:,i));
% end
% figure(fig1)
% plot(estPos(1,:),estPos(2,:),'.c','DisplayName','CV High PN')
% hold on
% axis([12.9610 12.9690 4.8880 4.8940] * 1e6)
% figure(fig2)
% plot((1:numSteps)*dt,dist,'c','DisplayName', 'CV High PN')



initialCovariance = diag([1,1e4,1,1e4,10]); 
initialGuess = [measPos(1,1); -1; measPos(2,1); -1; 0];
ctekf1 = trackingEKF(@constturn,@ctmeas,initialGuess, ...
    'StateTransitionJacobianFcn',@constturnjac, ...
    'MeasurementJacobianFcn',@ctmeasjac, ...
    'StateCovariance', initialCovariance, ...
    'HasAdditiveProcessNoise',false, ...
    'ProcessNoise', eye(3));

estimateStates = NaN(5, numSteps);
estimateStates(:,1) = ctekf1.State;
for i=2:numSteps
    predict(ctekf1,dt);
    dist(i) = distance(ctekf1,truePos(:,i));
    estimateStates(:,i) = correct(ctekf1,measPos(:,i));
end

figure(fig1);
plot(estimateStates(1,:),estimateStates(3,:),'.c','DisplayName','Constant Turn-Rate');
hold on

figure(fig2);
plot((1:numSteps)*dt,dist,'c','DisplayName', 'Constant Turn-Rate')

%%%%%%%%%%%%Constant turn-rate with high process noise%%%%%%%%%%%%%
% ctekf2 = trackingEKF(@constturn,@ctmeas,initialGuess, ...
%     'StateTransitionJacobianFcn',@constturnjac, ...
%     'MeasurementJacobianFcn',@ctmeasjac, ...
%     'StateCovariance', initialCovariance, ...
%     'HasAdditiveProcessNoise',false, ...
%     'ProcessNoise',  diag([50,50,1]));
% 
% estimateStates = NaN(5, numSteps);
% estimateStates(:,1) = ctekf2.State;
% for i=2:numSteps
%     predict(ctekf2,dt);
%     estimateStates(:,i) = correct(ctekf2,measPos(:,i));
% end
% figure(fig3);
% hold on
% plot(estimateStates(1,:),estimateStates(3,:),'.c','DisplayName','CT High PN');
% hold on




imm = trackingIMM('TransitionProbabilities', 0.99); % The default IMM has all three models
% Initialize the state and state covariance in terms of the first model
initialState = positionSelector' * measPos(:,1);
initialCovariance = diag([1,1e4,1,1e4,1,1e4]);   % Velocity is not measured
initialize(imm, initialState, initialCovariance);
dist = zeros(1,numSteps);
estPos = zeros(3,numSteps);
modelProbs = zeros(3,numSteps);
modelProbs(:,1) = imm.ModelProbabilities;
for i = 2:size(measPos,2)
    predict(imm, dt);
    dist(i) = distance(imm,truePos(:,i)); % Distance from true position
    estPos(:,i) = positionSelector * correct(imm, measPos(:,i));
    modelProbs(:,i) = imm.ModelProbabilities;
end


figure(fig1)
plot(estPos(1,:),estPos(2,:),'.r','DisplayName','IMM')

figure(fig2)
hold on
plot((1:numSteps)*dt,dist,'r','DisplayName', 'IMM')



saveas(fig1,'C:\Users\86188\Pictures\trakingpics\case_24_tracking', 'fig')
saveas(fig2, 'C:\Users\86188\Pictures\trakingpics\case_24_dist', 'fig')
saveas(fig1,'C:\Users\86188\Pictures\trakingpics\case_24_tracking', 'png')
saveas(fig2, 'C:\Users\86188\Pictures\trakingpics\case_24_dist', 'png')
