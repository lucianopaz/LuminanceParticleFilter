load tesis_figure_data3
if ~exist('t','var')
    t = (0:size(value,1)-1)*0.02;
end
dt = t(2)-t(1);
if ~exist('g','var')
    g = linspace(0,1,size(value,2));
end


figure('position',[100   100   950   600])
a1 = subplot(221);
a2 = subplot(222);
a3 = subplot(223);
a4 = subplot(224);

set(gcf,'currentAxes',a1)
imagesc(t,g,value')
cbar = colorbar;
ylabel(cbar,'$\tilde{V}$','interpreter','latex')
set(gca,'ydir','normal')
xlabel('tiempo [s]')
ylabel('g')
set(gca,'xlim',[0,20])


set(gcf,'currentAxes',a2)
plot(t,gb','linewidth',2)
xlabel('tiempo [s]')
ylabel('g')
set(gca,'xlim',[0,20])

set(gcf,'currentAxes',a4)
plot(t,xb','linewidth',2)
RandStream.setDefaultStream(RandStream('mt19937ar','seed',345678916));
samples = cumsum(sqrt(50/0.04)*dt*randn(length(t),4),1);
rt_ind = nan(1,size(samples,2));
dec_ind = rt_ind;
for i = 1:size(samples,2)
    temp1 = find(samples(:,i)>=xb(1,:)',1);
    temp2 = find(samples(:,i)<=xb(2,:)',1);
    if ~isempty(temp1) && isempty(temp2)
        rt_ind(i) = temp1;
        dec_ind(i) = 1;
        val = xb(1,temp1);
    elseif isempty(temp1) && ~isempty(temp2)
        rt_ind(i) = temp2;
        dec_ind(i) = 2;
        val = xb(2,temp2);
    else
        [rt_ind(i),dec_ind(i)] = min([temp1,temp2]);
        val = xb(dec_ind(i),rt_ind(i));
    end
    if ~isnan(rt_ind(i))
        samples(rt_ind(i)+1:end,i) = nan;
        samples(rt_ind(i),i) = val;
    end
end
colors = [0,0,0;0,0,1;0,0.5,0];
hold on
for i = 0:2
    if any(dec_ind==i)
        plot(t,samples(:,dec_ind==i),'color',colors(i+1,:));
    end
end
hold off
xlabel('tiempo [s]')
ylabel('g')
set(gca,'xlim',[0,20])


set(gcf,'currentAxes',a3)
plot(g,value(end,:),'--k','linewidth',3)
hold all
inds = [1,101,176,401];
colors = othercolor('RdYlGn10',length(inds)); colors = colors(end:-1:1,:);
for i = 1:length(inds)
    plot(g,v_explore(inds(i),:)','color',colors(i,:),'linewidth',2)
end
xl = gb(2,inds(end));
yl = value(inds(end),xl==g);
xu = gb(1,inds(end));
yu = value(inds(end),xu==g);
xlim = get(gca,'xlim');
ylim = get(gca,'ylim'); ylim(1) = -0.8;
plot([xl,xl],[ylim(1),yl],'--k')
plot([xu,xu],[ylim(1),yu],'--k')
xarea = g(xl<=g & xu>=g);
yarea = [value(end,xl<=g & xu>=g),repmat(ylim(1),1,2)];
xarea = [xarea, xu, xl];
patch(xarea,yarea,[0.6,0.6,0.6],'edgecolor','none','zdata',-4*ones(size(xarea)))
set(gca,'ylim',ylim,'layer','top')
hold off
xlabel('x(t)')
ylabel('$\tilde{V}$','interpreter','latex')


set(findall(gcf,'type','text'),'fontSize',15)
set(findobj(gcf,'type','axes','-and','tag',''),'fontsize',12)
set(findobj(gcf,'type','text','-and','tag','threshold'),'fontsize',18)
set(findobj(gcf,'type','axes','-and','tag','legend'),'fontsize',12)

place_title('A','northwest',a1,'fontsize',20,'displacement',[30,-5])
place_title('B','northwest',a2,'fontsize',20,'displacement',[30,-5])
place_title('C','northwest',a3,'fontsize',20,'displacement',[30,-5])
place_title('D','northwest',a4,'fontsize',20,'displacement',[30,-5]);

set(gcf,'color','w')
% plot2svg('bayesdec.svg',gcf);