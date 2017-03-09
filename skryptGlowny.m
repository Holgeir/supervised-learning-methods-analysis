clc;
clearvars;
close all;


% load('E-GEOD-3494A.mat');
% load('E-MTAB-5061.mat');

% c = cvpartition(status,'HoldOut',0.25);
% idx = training(c);


% dane_uczace = dane(:,idx);
% status_uczacy = status(idx);
% dane_testowe = dane(:,~idx);
% status_testowy = status(~idx);
% save('daneGEOD.mat', 'dane_uczace', 'status_uczacy', 'dane_testowe', 'status_testowy');
% save('daneMTAB.mat', 'dane_uczace', 'status_uczacy', 'dane_testowe', 'status_testowy');

% load('daneGEOD.mat', 'dane_uczace', 'status_uczacy', 'dane_testowe', 'status_testowy');
load('daneMTAB.mat', 'dane_uczace', 'status_uczacy', 'dane_testowe', 'status_testowy');

k = 5;
prog = 0.5;
liczba_cech = 500;
options = statset('MaxIter',100000,'Display','off', 'UseParallel', true);


tic;
[blad_dac, idx_m_dac, ocena_jakosci_dac, dokladnosc_dac, czulosc_dac, specyficznosc_dac, blad_wazony_dac] = k_fold_dac(dane_uczace, status_uczacy, k, liczba_cech, prog);
[t_dac] = toc;

for i=1:liczba_cech
     pd = fitdist(blad_wazony_dac(:,i),'Normal');
     ci = paramci(pd);
     [przedzial_ufnosci_blad_wazony_dac(:,i)] = ci(:,1);
end


for i=1:liczba_cech
    pd = fitdist(czulosc_dac(:,i),'Normal');
    ci = paramci(pd);
    [przedzial_ufnosci_czulosc_dac(:,i)] = ci(:,1);
end

for i=1:liczba_cech
    pd = fitdist(specyficznosc_dac(:,i),'Normal');
    ci = paramci(pd);
    [przedzial_ufnosci_specyficznosc_dac(:,i)] = ci(:,1);
end

[blad_wazony_dac] = sum(blad_wazony_dac)/k;
[specyficznosc_dac] = sum(specyficznosc_dac)/k;
[czulosc_dac] = sum(czulosc_dac)/k;


figure(1);
subplot(3,1,1);
plot(blad_wazony_dac);
title('B³¹d wa¿ony');
xlabel('Liczba cech');
ylabel('B³¹d wa¿ony');
hold on
plot(przedzial_ufnosci_blad_wazony_dac(1,:), ':');
plot(przedzial_ufnosci_blad_wazony_dac(2,:), ':');
subplot(3,1,2);
plot(czulosc_dac);
title('Czu³oœæ');
xlabel('Liczba cech');
ylabel('Czu³oœæ');
hold on
plot(przedzial_ufnosci_czulosc_dac(1,:), ':');
plot(przedzial_ufnosci_czulosc_dac(2,:), ':');
subplot(3,1,3);
plot(specyficznosc_dac);
title('Specyficznoœæ');
xlabel('Liczba cech');
ylabel('Specyficznoœæ');
hold on
plot(przedzial_ufnosci_specyficznosc_dac(1,:), ':');
plot(przedzial_ufnosci_specyficznosc_dac(2,:), ':');
savefig('Klasyfikator analizy dyskryminacyjnej GEOD.fig');

tic;
[blad_lr, idx_m_lr, ocena_jakosci_lr, dokladnosc_lr, czulosc_lr, specyficznosc_lr, blad_wazony_lr, dokladnosc_lr_sr, czulosc_lr_sr, specyficznosc_lr_sr, blad_wazony_lr_sr] = k_fold_lr(dane_uczace, status_uczacy, k, liczba_cech, prog);
[t_lr] = toc;

% for i=1:k
%     dokladnosc_lr_p(i) = dokladnosc_lr(i,15,14);
%     czulosc_lr_p(i) = czulosc_lr(i,15,14);
%     specyficznosc_lr_p(i) = specyficznosc_lr(i,15,14);
%     blad_wazony_lr_p(i) = blad_wazony_lr(i,15,14);
% end
% 
% for i=1:k
%      pd = fitdist(blad_wazony_lr_p','Normal');
%      ci = paramci(pd);
%      [przedzial_ufnosci_blad_wazony_lr_p(:,i)] = ci(:,1);
% end
% 
% 
% for i=1:k
%     pd = fitdist(czulosc_lr_p','Normal');
%     ci = paramci(pd);
%     [przedzial_ufnosci_czulosc_lr_p(:,i)] = ci(:,1);
% end
% 
% for i=1:k
%     pd = fitdist(specyficznosc_lr_p','Normal');
%     ci = paramci(pd);
%     [przedzial_ufnosci_specyficznosc_lr_p(:,i)] = ci(:,1);
% end

[blad_wazony_lr_sr] = reshape(blad_wazony_lr_sr,[size(blad_wazony_lr_sr,2),size(blad_wazony_lr_sr,3)])';
[czulosc_lr_sr] = reshape(czulosc_lr_sr,[size(czulosc_lr_sr,2),size(czulosc_lr_sr,3)])';
[specyficznosc_lr_sr] = reshape(specyficznosc_lr_sr,[size(specyficznosc_lr_sr,2),size(specyficznosc_lr_sr,3)])';


figure(2);
subplot(3,1,1);
contourf(1:liczba_cech, 0.1:0.05:0.9, blad_wazony_lr_sr); colorbar;
title('B³¹d wa¿ony');
xlabel('Liczba cech');
ylabel('Próg');
subplot(3,1,2);
contourf(1:liczba_cech, 0.1:0.05:0.9, czulosc_lr_sr); colorbar;
title('Czu³oœæ');
xlabel('Liczba cech');
ylabel('Próg');
subplot(3,1,3);
contourf(1:liczba_cech, 0.1:0.05:0.9, specyficznosc_lr_sr); colorbar;
title('Specyficznoœæ');
xlabel('Liczba cech');
ylabel('Próg');
savefig('Regresja logistyczna GEOD.fig');


tic;
[blad_svm, idx_m_svm, ocena_jakosci_svm, dokladnosc_svm, czulosc_svm, specyficznosc_svm, blad_wazony_svm, dokladnosc_svm_sr, czulosc_svm_sr, specyficznosc_svm_sr, blad_wazony_svm_sr] = k_fold_svm(dane_uczace, status_uczacy, k, liczba_cech, prog);
[t_svm] = toc;

% for i=1:k
%     dokladnosc_svm_p(i) = dokladnosc_svm(i,36,1);
%     czulosc_svm_p(i) = czulosc_svm(i,36,1);
%     specyficznosc_svm_p(i) = specyficznosc_svm(i,36,1);
%     blad_wazony_svm_p(i) = blad_wazony_svm(i,36,1);
% end
% 
% for i=1:k
%      pd = fitdist(blad_wazony_svm_p','Normal');
%      ci = paramci(pd);
%      [przedzial_ufnosci_blad_wazony_svm_p(:,i)] = ci(:,1);
% end
% 
% 
% for i=1:k
%     pd = fitdist(czulosc_svm_p','Normal');
%     ci = paramci(pd);
%     [przedzial_ufnosci_czulosc_svm_p(:,i)] = ci(:,1);
% end
% 
% for i=1:k
%     pd = fitdist(specyficznosc_svm_p','Normal');
%     ci = paramci(pd);
%     [przedzial_ufnosci_specyficznosc_svm_p(:,i)] = ci(:,1);
% end

[blad_wazony_svm_sr] = reshape(blad_wazony_svm_sr,[size(blad_wazony_svm_sr,2),size(blad_wazony_svm_sr,3)])';
[czulosc_svm_sr] = reshape(czulosc_svm_sr,[size(czulosc_svm_sr,2),size(czulosc_svm_sr,3)])';
[specyficznosc_svm_sr] = reshape(specyficznosc_svm_sr,[size(specyficznosc_svm_sr,2),size(specyficznosc_svm_sr,3)])';

figure(3);
subplot(3,1,1);
contourf(1:liczba_cech, -2:0.2:0, blad_wazony_svm_sr); colorbar;
title('B³¹d wa¿ony');
xlabel('Liczba cech');
ylabel('Wspó³czynnik kary [10^]','Interpreter','none');
subplot(3,1,2);
contourf(1:liczba_cech, -2:0.2:0, czulosc_svm_sr); colorbar;
title('Czu³oœæ');
xlabel('Liczba cech');
ylabel('Wspó³czynnik kary [10^]','Interpreter','none');
subplot(3,1,3);
contourf(1:liczba_cech, -2:0.2:0, specyficznosc_svm_sr); colorbar;
title('Specyficznoœæ');
xlabel('Liczba cech');
ylabel('Wspó³czynnik kary [10^]','Interpreter','none');
savefig('SVM GEOD.fig');



idx_opt_dac = idx_m_dac(1:11,:);
idx_opt_dac = idx_opt_dac(:);
tbl = tabulate(idx_opt_dac);
tbl2 = tbl(find(tbl(:,2)),1:3);
tbl2 = sortrows(tbl2, -2);
idx_opt_dac = tbl2(:,1);
idx_opt_dac = idx_opt_dac(1:11);
idx_count_dac = 100*tbl2(1:11,2)/k;

idx_opt_lr = idx_m_lr(1:11,:);
idx_opt_lr = idx_opt_lr(:);
tbl = tabulate(idx_opt_lr);
tbl2 = tbl(find(tbl(:,2)),1:3);
tbl2 = sortrows(tbl2, -2);
idx_opt_lr = tbl2(:,1);
idx_opt_lr = idx_opt_lr(1:11);
idx_count_lr = 100*tbl2(1:11,2)/k;

idx_opt_svm = idx_m_svm(1:11,:);
idx_opt_svm = idx_opt_svm(:);
tbl = tabulate(idx_opt_svm);
tbl2 = tbl(find(tbl(:,2)),1:3);
tbl2 = sortrows(tbl2, -2);
idx_opt_svm = tbl2(:,1);
idx_opt_svm = idx_opt_svm(1:11);
idx_count_svm = 100*tbl2(1:11,2)/k;


prog = 0.5;

model = fitcdiscr(dane_uczace(idx_opt_dac,:)', status_uczacy);
predykcja = predict(model, dane_testowe(idx_opt_dac,:)');
klasa = predykcja > prog;
[blad_test_dac] = 100 * sum(xor(klasa,status_testowy'))/length(status_testowy);

ocena_jakosci_dac_test = zeros(2,2);
    for a=1:length(status_testowy)
        if (klasa(a,1) == 1 && status_testowy(1,a) == 1)
                ocena_jakosci_dac_test(1,1) = ocena_jakosci_dac_test(1,1) + 1;
            elseif (klasa(a,1) == 0 && status_testowy(1,a) == 0)
                ocena_jakosci_dac_test(2,2) = ocena_jakosci_dac_test(2,2) + 1;
            elseif (klasa(a,1) == 1 && status_testowy(1,a) == 0)
                ocena_jakosci_dac_test(1,2) = ocena_jakosci_dac_test(1,2) + 1;
            else 
                ocena_jakosci_dac_test(2,1) = ocena_jakosci_dac_test(2,1) + 1;
        end
    end
    
[dokladnosc_dac_test] = (ocena_jakosci_dac_test(1,1) + ocena_jakosci_dac_test(2,2)) / (ocena_jakosci_dac_test(1,1) + ocena_jakosci_dac_test(1,2) + ocena_jakosci_dac_test(2,1) + ocena_jakosci_dac_test(2,2));
[czulosc_dac_test] = ocena_jakosci_dac_test(1,1) / (ocena_jakosci_dac_test(1,1) + ocena_jakosci_dac_test(2,1));
[specyficznosc_dac_test] = ocena_jakosci_dac_test(2,2) / (ocena_jakosci_dac_test(2,2) + ocena_jakosci_dac_test(1,2));
[blad_wazony_dac_test] = 1 - ((czulosc_dac_test + specyficznosc_dac_test) / 2);

boxconstraint = -2;

model = svmtrain(dane_uczace(idx_opt_dac,:)', status_uczacy, 'boxconstraint', 10^boxconstraint, 'options', options);
predykcja = svmclassify(model, dane_testowe(idx_opt_dac,:)');
klasa = predykcja > prog;
[blad_test_svm] = 100 * sum(xor(klasa,status_testowy'))/length(status_testowy);

ocena_jakosci_svm_test = zeros(2,2);
    for a=1:length(status_testowy)
        if (klasa(a,1) == 1 && status_testowy(1,a) == 1)
                ocena_jakosci_svm_test(1,1) = ocena_jakosci_svm_test(1,1) + 1;
            elseif (klasa(a,1) == 0 && status_testowy(1,a) == 0)
                ocena_jakosci_svm_test(2,2) = ocena_jakosci_svm_test(2,2) + 1;
            elseif (klasa(a,1) == 1 && status_testowy(1,a) == 0)
                ocena_jakosci_svm_test(1,2) = ocena_jakosci_svm_test(1,2) + 1;
            else 
                ocena_jakosci_svm_test(2,1) = ocena_jakosci_svm_test(2,1) + 1;
        end
    end
    
[dokladnosc_svm_test] = (ocena_jakosci_svm_test(1,1) + ocena_jakosci_svm_test(2,2)) / (ocena_jakosci_svm_test(1,1) + ocena_jakosci_svm_test(1,2) + ocena_jakosci_svm_test(2,1) + ocena_jakosci_svm_test(2,2));
[czulosc_svm_test] = ocena_jakosci_svm_test(1,1) / (ocena_jakosci_svm_test(1,1) + ocena_jakosci_svm_test(2,1));
[specyficznosc_svm_test] = ocena_jakosci_svm_test(2,2) / (ocena_jakosci_svm_test(2,2) + ocena_jakosci_svm_test(1,2));
[blad_wazony_svm_test] = 1 - ((czulosc_svm_test + specyficznosc_svm_test) / 2);
            

prog = 0.75; 

model = glmfit(dane_uczace(idx_opt_lr,:)', status_uczacy', 'binomial');
predykcja = glmval(model, dane_testowe(idx_opt_lr,:)', 'logit');
klasa = predykcja > prog;
[blad_test_lr] = 100 * sum(xor(klasa,status_testowy'))/length(status_testowy);

ocena_jakosci_lr_test = zeros(2,2);
    for a=1:length(status_testowy)
        if (klasa(a,1) == 1 && status_testowy(1,a) == 1)
                ocena_jakosci_lr_test(1,1) = ocena_jakosci_lr_test(1,1) + 1;
            elseif (klasa(a,1) == 0 && status_testowy(1,a) == 0)
                ocena_jakosci_lr_test(2,2) = ocena_jakosci_lr_test(2,2) + 1;
            elseif (klasa(a,1) == 1 && status_testowy(1,a) == 0)
                ocena_jakosci_lr_test(1,2) = ocena_jakosci_lr_test(1,2) + 1;
            else 
                ocena_jakosci_lr_test(2,1) = ocena_jakosci_lr_test(2,1) + 1;
        end
    end
    
[dokladnosc_lr_test] = (ocena_jakosci_lr(1,1) + ocena_jakosci_lr(2,2)) / (ocena_jakosci_lr(1,1) + ocena_jakosci_lr(1,2) + ocena_jakosci_lr(2,1) + ocena_jakosci_lr(2,2));
[czulosc_lr_test] = ocena_jakosci_lr(1,1) / (ocena_jakosci_lr(1,1) + ocena_jakosci_lr(2,1));
[specyficznosc_lr_test] = ocena_jakosci_lr(2,2) / (ocena_jakosci_lr(2,2) + ocena_jakosci_lr(1,2));
[blad_wazony_lr_test] = 1 - ((czulosc_lr_test + specyficznosc_lr_test) / 2);










