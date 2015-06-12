function fval = simple_objective_function(x,RT,dprime,T)
    sim_RT = zeros(size(RT));
    threshold_passed = abs(dprime)>=x(1);
    for i = 1:size(dprime,1)
        ind = find(threshold_passed(i,:),1);
        if ~isempty(ind)
            sim_RT(i) = T(ind);
        else
            sim_RT(i) = T(end);
        end
    end
    fval = sum((RT-sim_RT-x(2)).^2);
end