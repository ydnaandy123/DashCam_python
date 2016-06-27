D = dir('info_3d');
% for i=1:length(D.name)
%     D.name(i)
% end

fname = 'namelist.json';
fp = fopen(fname, 'w');
fprintf(fp, '[');
fprintf(fp, '"%s",', D.name);
fprintf(fp, ']');
fclose(fp);