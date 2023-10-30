clear all; clc; close all;

%%

filenames = {'sentence1','sentence2','sentence3','sentence4','sentence5','sentence6','sentence7','sentence8','sentence9'}
savenames = {'1','2','3','4','5','6','7','8','9'}
subnums = [1]
datadir = 'data'
for subnum = subnums
    subname = ['sub' num2str(subnum)];
    
    fs_imu = 50
    fs_snd = 44100;
    saveDir_imu = [datadir '\epo\sub' num2str(subnum) '\imu\'];
    if ~isfolder(saveDir_imu)
        mkdir(saveDir_imu);
    end
    saveDir_snd = [datadir '\epo\sub' num2str(subnum) '\snd\'];
    if ~isfolder(saveDir_snd)
        mkdir(saveDir_snd);
    end
    rest = 0;
    beep_t = 1;
    end_t = 0.1;
    for k = 1:length(filenames)
        clear snd imu emg mrk
        name = filenames{k};
        savename = savenames{k};
        sub_folder = [datadir '/org/' subname '/'];
        load([sub_folder name '.mat']);
        trig = data.trigger_time(:,1);
        session_num = trig(1);
        trig(1) = [];
        trig_s = trig(trig<100);
        trig_e = trig(trig>100);
        trig_s(trig_e == 200) = [];
        ntrial = length(trig_s);
        if(ntrial>30)
                keyboard
        end
        snd.fs = fs_snd/2;
        imu.fs = fs_imu;
%         session_num = data.session;
        
        for i = 1:ntrial
            i;
            snd.x{i} = data.sound{i}.data;
            snd.x{i} = resample(snd.x{i},1,2); %%down sampling to 22050Hz
            
            snd_length = length(snd.x{i});
            snd.time{i} = [1:snd_length]/snd.fs;
                 
            imu_start = data.trigger(i,2);
            imu_end = data.trigger(i,3);

            imu.x{i} = data.imu{i}.data;
            imu.time{i} = [1:length(imu.x{i})]/imu.fs;

            sig_len = floor(min([snd.time{i}(end),imu.time{i}(end)])*10)/10; %% 둘 중 더 짧은 data 기준으로 epoching
            snd_temp = snd.x{i};
            snd.x{i} = snd.x{i}(1+floor(beep_t*snd.fs):floor(sig_len*snd.fs) - floor(end_t*snd.fs),:);
            snd.time{i} = snd.time{i}(1+floor(beep_t*snd.fs):floor(sig_len*snd.fs)- floor(end_t*snd.fs));
            
            imu.x{i} = imu.x{i}(1+floor(beep_t*imu.fs):floor(sig_len*imu.fs)- floor(end_t*imu.fs),:);
            imu.time{i} = imu.time{i}(1+floor(beep_t*imu.fs):floor(sig_len*imu.fs)- floor(end_t*imu.fs));
        end
        imu.session = session_num;
        snd.session = (session_num-200)*100 + trig_s;
        imu.trigger = (session_num-200)*100 + trig_s;
        
        save([saveDir_imu '\epo' savename '.mat'],'imu');
        save([saveDir_snd '\epo' savename '.mat'],'snd');
    end
end