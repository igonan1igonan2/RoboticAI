clear all

Subs = [1]
sessions = {'1','2','3','4','5','6','7','8','9'}%,'num'}
min_db = log(1e-4)
datadir = 'data'
mel_feat = 'nr'
for s = Subs
    nt = 1;    
    load([datadir '\feat\sub' num2str(s) '\feat_' mel_feat '.mat']);

    nTrial = length(sig)
    for dat = 1:nTrial
        [len,ch] = size(sig{dat});
        
        pad_idx(dat,1) = len;
        
        sig_mask(dat,1:len) = true;
        sig_mask(dat,len+1:max_len)=false;
        sig_pad(dat,:,:) = cat(1,sig{dat},zeros(max_len-len,ch));
        
        sig_flip = flip(sig{dat})*-1;
        sig_flip_pad(dat,:,:) = cat(1,sig_flip,zeros(max_len-len,ch));
        
        [len,feat] = size(mels{dat});
        mels_mask(dat,1:len) = true;
        mels_mask(dat,len+1:max_len)=false;
        mels_pad(dat,:,:) = cat(1,mels{dat},ones(max_len-len,feat)*min_db);
        
        mels_flip = flip(mels{dat});
        mels_flip_pad(dat,:,:) = cat(1,mels_flip,ones(max_len-len,feat)*min_db);

    end
    
    save([datadir '\feat\sub' num2str(s) '\sig_flip_padd.mat'],'sig_pad','sig_flip_pad','max_len','pad_idx','sig_mask')
    save([datadir '\feat\sub' num2str(s) '\mels_' mel_feat '_flip_padd.mat'],'mels_pad','mels_flip_pad','max_len','pad_idx','mels_mask')    
    clear sig_pad mels_pad sig_flip_pad mels_flip_pad pad_idx sig_mask mels_mask
end


