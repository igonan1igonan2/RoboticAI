used_Sub = [1]

for sub = used_Sub
    trig =[];
    for sess  = 1:9
        load(['data/epo/sub' num2str(sub) '/imu/epo' num2str(sess) '.mat']);
        trig = cat(1,trig,imu.trigger);
        
    end
    sess = floor(trig/100);
    trig_idx = (sess-1)*30 + trig - sess*100;
    save_dir = ['trig/sub' num2str(sub) ]
    if ~isfolder(save_dir)
        mkdir(save_dir)
    end
    save([save_dir '/trig.mat'],'trig','trig_idx');
end