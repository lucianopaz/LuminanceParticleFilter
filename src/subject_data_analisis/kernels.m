function [decision_kernel,confidence_kernel,decision_kernel_std,confidence_kernel_std] =...
    kernels(tfluct,dfluct,selection,confidence,is_binary_confidence,locked_on_onset,RT_ind)

    if nargin<6
        locked_on_onset = true;
        if nargin<5
            is_binary_confidence = true;
        end
    end

    if ~locked_on_onset
        target_T_dec = nan(size(tfluct,1),2*size(tfluct,2)-1);
        distractor_T_dec = nan(size(dfluct,1),2*size(dfluct,2)-1);
        % Center the luminance fluctuation at response time in the middle of the
        % vector
        for i = 1:size(tfluct,1)
            target_T_dec(i,size(tfluct,2)+1-RT_ind(i):2*size(tfluct,2)-RT_ind(i)) = tfluct(i,:);
            distractor_T_dec(i,size(dfluct,2)+1-RT_ind(i):2*size(dfluct,2)-RT_ind(i)) = dfluct(i,:);
        end
        tfluct = target_T_dec;
        dfluct = distractor_T_dec;
    end

    hsel = nan(size(tfluct));
    hnotsel = nan(size(tfluct));
    lsel = nan(size(tfluct));
    lnotsel = nan(size(tfluct));
    hits = selection==1;
    misses = selection==2;
    if is_binary_confidence
        high_confidence = confidence==2;
        low_confidence = confidence==1;
        if any(hits & high_confidence)
            hsel(hits & high_confidence,:) = tfluct(hits & high_confidence,:);
            hnotsel(hits & high_confidence,:) = dfluct(hits & high_confidence,:);
        end
        if any(hits & low_confidence)
            lsel(hits & low_confidence,:) = tfluct(hits & low_confidence,:);
            lnotsel(hits & low_confidence,:) = dfluct(hits & low_confidence,:);
        end
        if any(misses & high_confidence)
            hsel(misses & high_confidence,:) = dfluct(misses & high_confidence,:);
            hnotsel(misses & high_confidence,:) = tfluct(misses & high_confidence,:);
        end
        if any(misses & low_confidence)
            lsel(misses & low_confidence,:) = dfluct(misses & low_confidence,:);
            lnotsel(misses & low_confidence,:) = tfluct(misses & low_confidence,:);
        end
        if ~locked_on_onset
            hsel(:,sum(~isnan(hsel))<0.5*sum(~all(isnan(hsel),2))) = nan;
            hnotsel(:,sum(~isnan(hnotsel))<0.5*sum(~all(isnan(hnotsel),2))) = nan;
            lsel(:,sum(~isnan(lsel))<0.5*sum(~all(isnan(lsel),2))) = nan;
            lnotsel(:,sum(~isnan(lnotsel))<0.5*sum(~all(isnan(lnotsel),2))) = nan;
        end
        decision_kernel = [nanmean(cat(1,hsel,lsel),1);nanmean(cat(1,hnotsel,lnotsel),1)];
        confidence_kernel = [nanmean(hsel,1)-nanmean(lsel,1);nanmean(hnotsel,1)-nanmean(lnotsel,1)];
        if nargout>3
            decision_kernel_std = [nanstd(cat(1,hsel,lsel),1);nanstd(cat(1,hnotsel,lnotsel),1)]/sqrt(size(tfluct,1));
            confidence_kernel_std = [nanstd(hsel)/sqrt(sum(~all(isnan(hsel),2)))+nanstd(lsel)/sqrt(sum(~all(isnan(lsel),2)));...
                                     nanstd(hnotsel)/sqrt(sum(~all(isnan(hnotsel),2)))+nanstd(lnotsel)/sqrt(sum(~all(isnan(lnotsel),2)))];
        end
    else
        if any(hits)
            hsel(hits,:) = tfluct(hits,:).*repmat(confidence(hits),1,size(tfluct,2));
            lsel(hits,:) = tfluct(hits,:).*repmat((1-confidence(hits)),1,size(tfluct,2));
            hnotsel(hits,:) = dfluct(hits,:).*repmat(confidence(hits),1,size(tfluct,2));
            lnotsel(hits,:) = dfluct(hits,:).*repmat((1-confidence(hits)),1,size(tfluct,2));
        end
        if any(misses)
            hsel(misses,:) = dfluct(misses,:).*repmat(confidence(misses),1,size(tfluct,2));
            lsel(misses,:) = dfluct(misses,:).*repmat((1-confidence(misses)),1,size(tfluct,2));
            hnotsel(misses,:) = tfluct(misses,:).*repmat(confidence(misses),1,size(tfluct,2));
            lnotsel(misses,:) = tfluct(misses,:).*repmat((1-confidence(misses)),1,size(tfluct,2));
        end
        if ~locked_on_onset
            hsel(:,sum(~isnan(hsel))<0.5*sum(~all(isnan(hsel),2))) = nan;
            hnotsel(:,sum(~isnan(hnotsel))<0.5*sum(~all(isnan(hnotsel),2))) = nan;
            lsel(:,sum(~isnan(lsel))<0.5*sum(~all(isnan(lsel),2))) = nan;
            lnotsel(:,sum(~isnan(lnotsel))<0.5*sum(~all(isnan(lnotsel),2))) = nan;
        end
        decision_kernel = [nanmean(hsel+lsel,1);nanmean(hnotsel+lnotsel,1)];
        confidence_kernel = [nanmean(hsel,1)-nanmean(lsel,1);nanmean(hnotsel,1)-nanmean(lnotsel,1)];
        if nargout>3
            decision_kernel_std = [nanstd(hsel+lsel,1);nanstd(hnotsel+lnotsel,1)]/sqrt(size(tfluct,1));
            confidence_kernel_std = [nanstd(hsel)/sqrt(sum(~all(isnan(hsel),2)))+nanstd(lsel)/sqrt(sum(~all(isnan(lsel),2)));...
                                 nanstd(hnotsel)/sqrt(sum(~all(isnan(hnotsel),2)))+nanstd(lnotsel)/sqrt(sum(~all(isnan(lnotsel),2)))];
        end
    end
end