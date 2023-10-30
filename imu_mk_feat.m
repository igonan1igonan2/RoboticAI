clear all
Subs = [1]
sessions = [1,2,3,4,5,6,7,8,9]
mel_feat = 'nr';
datadir = 'data';
for s = Subs
    nt = 1;
    max_len = 0;
    for sess = sessions
        load([datadir '\epo\sub' num2str(s) '\imu\epo' num2str(sess)]);
         load([datadir '\feat\sub' num2str(s) '\snd.mat']);
    
        nTrial = length(imu.x);
        for dat = 1:nTrial
            sig{nt} = normalize(imu.x{dat}(2:end,:));
            if(length(sig{nt}) ~= size(eval(['mel.' mel_feat '_epo' num2str(nt)]),1))
                lenn = min([length(sig{nt}) size(eval(['mel. ' mel_feat '_epo' num2str(nt)]),1)]);                
                sig{nt} = sig{nt}(1:lenn,:);
                eval(['mels{' num2str(nt) '} = mel.' mel_feat '_epo' num2str(nt) '(1:lenn,:);']);
            else
                eval(['mels{' num2str(nt) '} = mel.' mel_feat '_epo' num2str(nt) ';']);
            end
            if max_len< length(sig{nt})
                   max_len = length(sig{nt});
            end
               nt = nt+1;
        end
        
    end
    max_len = ceil(max_len/8)*8;
    len_max(s) = max_len
     save([datadir '\feat\sub' num2str(s) '\feat_' mel_feat '.mat'],'sig','max_len','mels')
end
