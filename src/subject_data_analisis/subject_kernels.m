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
hsel = nan(size(target));
hnotsel = nan(size(target));
lsel = nan(size(target));
lnotsel = nan(size(target));
won_index = data(:,3)==1;
high_index = data(:,4)==2;
if any(won_index & high_index)
    hsel(won_index & high_index,:) = target(won_index & high_index,:)-repmat(data(won_index & high_index,1),1,size(target,2));
    hnotsel(won_index & high_index,:) = distractor(won_index & high_index,:)-50;
end
if any(won_index & ~high_index)
    lsel(won_index & ~high_index,:) = target(won_index & ~high_index,:)-repmat(data(won_index & ~high_index,1),1,size(target,2));
    lnotsel(won_index & ~high_index,:) = distractor(won_index & ~high_index,:)-50;
end
if any(~won_index & high_index)
    hsel(~won_index & high_index,:) = distractor(~won_index & high_index,:)-50;
    hnotsel(~won_index & high_index,:) = target(~won_index & high_index,:)-repmat(data(~won_index & high_index,1),1,size(target,2));
end
if any(won_index & ~high_index)
    lsel(~won_index & ~high_index,:) = distractor(~won_index & ~high_index,:)-50;
    lnotsel(~won_index & ~high_index,:) = target(~won_index & ~high_index,:)-repmat(data(~won_index & ~high_index,1),1,size(target,2));
end
% Decision and Confidence kernels locked on stimulus onset
decision_kernel = [nanmean(cat(1,hsel,lsel),1);nanmean(cat(1,hnotsel,lnotsel),1)];
confidence_kernel = [nanmean(hsel,1)-nanmean(lsel,1);nanmean(hnotsel,1)-nanmean(lnotsel,1)];
decision_kernel_std = [nanstd(cat(1,hsel,lsel),1);nanstd(cat(1,hnotsel,lnotsel),1)]/sqrt(size(target,1));
confidence_kernel_std = [nanstd(hsel)/sqrt(sum(~all(isnan(hsel),2)))+nanstd(lsel)/sqrt(sum(~all(isnan(lsel),2)));...
                         nanstd(hnotsel)/sqrt(sum(~all(isnan(hnotsel),2)))+nanstd(lnotsel)/sqrt(sum(~all(isnan(lnotsel),2)))];
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
T_dec = mod(RT,1e3);
[bla,T_dec_ind] = histc(T_dec,0:40:1000);
T_dec_ind(T_dec_ind==26) = 25;
target_T_dec = nan(size(target,1),2*size(target,2)-1);
distractor_T_dec = nan(size(distractor,1),2*size(distractor,2)-1);
% Center the luminance fluctuation at response time in the middle of the
% vector
for i = 1:size(target,1)
    target_T_dec(i,size(target,2)+1-T_dec_ind(i):2*size(target,2)-T_dec_ind(i)) = target(i,:);
    distractor_T_dec(i,size(distractor,2)+1-T_dec_ind(i):2*size(distractor,2)-T_dec_ind(i)) = distractor(i,:);
end

hsel = nan(size(target_T_dec));
hnotsel = nan(size(target_T_dec));
lsel = nan(size(target_T_dec));
lnotsel = nan(size(target_T_dec));
if any(won_index & high_index)
    hsel(won_index & high_index,:) = target_T_dec(won_index & high_index,:)-repmat(data(won_index & high_index,1),1,size(target_T_dec,2));
    hnotsel(won_index & high_index,:) = distractor_T_dec(won_index & high_index,:)-50;
end
if any(won_index & ~high_index)
    lsel(won_index & ~high_index,:) = target_T_dec(won_index & ~high_index,:)-repmat(data(won_index & ~high_index,1),1,size(target_T_dec,2));
    lnotsel(won_index & ~high_index,:) = distractor_T_dec(won_index & ~high_index,:)-50;
end
if any(~won_index & high_index)
    hsel(~won_index & high_index,:) = distractor_T_dec(~won_index & high_index,:)-50;
    hnotsel(~won_index & high_index,:) = target_T_dec(~won_index & high_index,:)-repmat(data(~won_index & high_index,1),1,size(target_T_dec,2));
end
if any(won_index & ~high_index)
    lsel(~won_index & ~high_index,:) = distractor_T_dec(~won_index & ~high_index,:)-50;
    lnotsel(~won_index & ~high_index,:) = target_T_dec(~won_index & ~high_index,:)-repmat(data(~won_index & ~high_index,1),1,size(target_T_dec,2));
end

% Ignore fluctuations at times with less than half the data
hinds = sum(~isnan(hsel))<0.5*sum(high_index);
linds = sum(~isnan(lsel))<0.5*sum(~high_index);
hsel(:,sum(~isnan(hsel))<0.5*sum(high_index)) = nan;
hnotsel(:,sum(~isnan(hnotsel))<0.5*sum(high_index)) = nan;
lsel(:,sum(~isnan(lsel))<0.5*sum(~high_index)) = nan;
lnotsel(:,sum(~isnan(lnotsel))<0.5*sum(~high_index)) = nan;

decision_kernel = [nanmean(cat(1,hsel,lsel),1);nanmean(cat(1,hnotsel,lnotsel),1)];
confidence_kernel = [nanmean(hsel,1)-nanmean(lsel,1);nanmean(hnotsel,1)-nanmean(lnotsel,1)];
decision_kernel_std = [nanstd(cat(1,hsel,lsel),1);nanstd(cat(1,hnotsel,lnotsel),1)]/sqrt(size(target,1));
confidence_kernel_std = [nanstd(hsel)/sqrt(sum(~all(isnan(hsel),2)))+nanstd(lsel)/sqrt(sum(~all(isnan(lsel),2)));...
                         nanstd(hnotsel)/sqrt(sum(~all(isnan(hnotsel),2)))+nanstd(lnotsel)/sqrt(sum(~all(isnan(lnotsel),2)))];
T = -1000:40:1000; T(T==-1000 | T==1000) = [];

T_decision = T(all(~isnan(decision_kernel),1));
decision_kernel(:,all(isnan(decision_kernel),1)) = [];
T_confidence = T(all(~isnan(confidence_kernel),1));
confidence_kernel(:,all(isnan(confidence_kernel),1)) = [];
decision_kernel_std(:,all(isnan(decision_kernel_std),1)) = [];
confidence_kernel_std(:,all(isnan(confidence_kernel_std),1)) = [];
T = -1000:40:1000; T(T==-1000 | T==1000) = [];
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
