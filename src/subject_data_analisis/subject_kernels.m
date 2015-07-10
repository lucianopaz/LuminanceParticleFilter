datadir = fullfile('~','Dropbox','DecisionConfidenceKernels','data');

subjects = unique_subjects(datadir);
[data,target,distractor] = load_stim_and_trial(subjects,6);

% Compute subject decision and confidence kernels
% Reduce the 4 luminous patches to a single patch.
target = squeeze(mean(target,3));
distractor = squeeze(mean(distractor,3));

% Performance divided in confidence
    % [correct high     , correct low     , correct with no confidence;...
    %  incorrect high   , incorrect low   , incorrect with no confidence;...
    %  no decision high , no decision low , no decision with no confidence];
number_of_selections = [sum(data(:,3)==1 & data(:,4)==2),sum(data(:,3)==1 & data(:,4)==1),0;...
                        sum(data(:,3)==0 & data(:,4)==2),sum(data(:,3)==0 & data(:,4)==1),0;...
                        0,0,0];

%% Subject response time histograms
RT = data(:,2);
[RT_hist,bins] = hist(RT,100);
RT_hist = RT_hist/sum(~isnan(RT));

RT_hist_subj = zeros(length(subjects),length(RT_hist));
for s = 1:length(subjects)
    RT_hist_subj(s,:) = hist(RT(data(:,6)==s),bins);
    RT_hist_subj(s,:) = RT_hist_subj(s,:)/sum(~isnan(RT) & data(:,6)==s);
end
figure('position',[100 100 1000 800])
plot(bins,RT_hist,'r','linewidth',3)
hold on
plot(bins,RT_hist_subj','k','linewidth',1)
hold off
xlabel('RT [ms]')
set(findall(gcf,'type','text'),'fontSize',18)
set(findobj(gcf,'type','axes','-and','tag',''),'fontsize',14)
set(findobj(gcf,'type','axes','-and','tag','legend'),'fontsize',14)

%% Compute decision and confidence kernels
tfluct = target-repmat(data(:,1),1,size(target,2));
dfluct = distractor-50;
selection = data(:,3); selection(data(:,3)~=1) = 2;
confidence = data(:,4);
[decision_kernel,confidence_kernel,decision_kernel_std,confidence_kernel_std] = ...
    kernels(tfluct,dfluct,selection,confidence);
T = 0:40:1000; T(T==1000)=[];
figure('position',[100 100 1000 800])
try
    subplot(1,2,1)
    errorzone(T,decision_kernel(1,:),decision_kernel_std(1,:),'--b','edgealpha',0,'facealpha',0.3);
    hold on
    errorzone(T,decision_kernel(2,:),decision_kernel_std(2,:),'--r','edgealpha',0,'facealpha',0.3);
    hold off
    xlabel('Time [ms]');
    title('Decision')
    subplot(1,2,2)
    errorzone(T,confidence_kernel(1,:),confidence_kernel_std(1,:),'--b','edgealpha',0,'facealpha',0.3);
    hold on
    errorzone(T,confidence_kernel(2,:),confidence_kernel_std(2,:),'--r','edgealpha',0,'facealpha',0.3);
    hold off
    xlabel('Time [ms]');
    title('Confidence')
catch
    subplot(1,2,1)
    plot(T,decision_kernel(1,:),'--b');
    hold on
    plot(T,decision_kernel(2,:),'--r');
    hold off
    xlabel('Time [ms]');
    title('Decision')
    subplot(1,2,2)
    plot(T,confidence_kernel(1,:),'--b');
    hold on
    plot(T,confidence_kernel(2,:),'--r');
    hold off
    xlabel('Time [ms]');
    title('Confidence')
end
set(findall(gcf,'type','text'),'fontSize',18)
set(findobj(gcf,'type','axes','-and','tag',''),'fontsize',14)
set(findobj(gcf,'type','axes','-and','tag','legend'),'fontsize',14)

%% Decision and Confidence kernels locked on response time
% T_dec = mod(RT,1e3);
% [bla,T_dec_ind] = histc(T_dec,0:40:1000);
% T_dec_ind(T_dec_ind==26) = 25;
% [decision_kernel,confidence_kernel,decision_kernel_std,confidence_kernel_std] = ...
%     kernels(tfluct,dfluct,selection,confidence,true,false,T_dec_ind);
% T = -1000:40:1000; T(T==-1000 | T==1000) = [];

T_dec = RT;
[bla,T_dec_ind] = histc(T_dec,0:40:5000);
T_dec_ind(T_dec_ind==126) = 125;
tfluct_ext = nan(size(tfluct,1),125); tfluct_ext(:,1:size(tfluct,2)) = tfluct;
dfluct_ext = nan(size(dfluct,1),125); dfluct_ext(:,1:size(dfluct,2)) = dfluct;
[decision_kernel,confidence_kernel,decision_kernel_std,confidence_kernel_std] = ...
    kernels(tfluct_ext,dfluct_ext,selection,confidence,true,false,T_dec_ind);
T = -4960:40:4960;

T_decision = T(all(~isnan(decision_kernel),1));
decision_kernel(:,all(isnan(decision_kernel),1)) = [];
T_confidence = T(all(~isnan(confidence_kernel),1));
confidence_kernel(:,all(isnan(confidence_kernel),1)) = [];
decision_kernel_std(:,all(isnan(decision_kernel_std),1)) = [];
confidence_kernel_std(:,all(isnan(confidence_kernel_std),1)) = [];
figure('position',[100 100 1000 800])
try
    subplot(1,2,1)
    errorzone(T_decision,decision_kernel(1,:),decision_kernel_std(1,:),'--b','edgealpha',0,'facealpha',0.3);
    hold on
    errorzone(T_decision,decision_kernel(2,:),decision_kernel_std(2,:),'--r','edgealpha',0,'facealpha',0.3);
    hold off
    xlabel('Time-RT [ms]');
    title('Decision')
    subplot(1,2,2)
    errorzone(T_confidence,confidence_kernel(1,:),confidence_kernel_std(1,:),'--b','edgealpha',0,'facealpha',0.3);
    hold on
    errorzone(T_confidence,confidence_kernel(2,:),confidence_kernel_std(2,:),'--r','edgealpha',0,'facealpha',0.3);
    hold off
    xlabel('Time-RT [ms]');
    title('Confidence')
catch
    subplot(1,2,1)
    plot(T_decision,decision_kernel(1,:),'--b');
    hold on
    plot(T_decision,decision_kernel(2,:),'--r');
    hold off
    xlabel('Time-RT [ms]');
    title('Decision')
    subplot(1,2,2)
    plot(T_confidence,confidence_kernel(1,:),'--b');
    hold on
    plot(T_confidence,confidence_kernel(2,:),'--r');
    hold off
    xlabel('Time-RT [ms]');
    title('Confidence')
end
set(findall(gcf,'type','text'),'fontSize',18)
set(findobj(gcf,'type','axes','-and','tag',''),'fontsize',14)
set(findobj(gcf,'type','axes','-and','tag','legend'),'fontsize',14)
