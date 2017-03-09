function [blad_dac, idx_m_dac, ocena_jakosci_dac, dokladnosc_dac, czulosc_dac, specyficznosc_dac, blad_wazony_dac] = k_fold_dac(dane_uczace, status_uczacy, k, liczba_cech, prog)

warning off

c = cvpartition(status_uczacy,'KFold',k);

blad_dac = zeros(k,50);
idx_m_dac = zeros(size(dane_uczace,1),k);
ocena_jakosci_dac = zeros(2,2);

for i=1:k
    idx = training(c,i);
    dane_treningowe = dane_uczace(:,idx);
    status_treningowy = status_uczacy(idx);
    dane_walidacyjne = dane_uczace(:,~idx);
    status_walidacyjny = status_uczacy(~idx);
    
    
	
	idx1 = rankfeatures(dane_treningowe, status_treningowy);
    idx_m_dac(:,i) = idx1;
    for b=1:liczba_cech
        disp(['DAC - iteracja: ' num2str(i) ', cecha: ' num2str(b)]);

        model = fitcdiscr(dane_treningowe(idx1(1:b),:)', status_treningowy);
        predykcja = predict(model, dane_walidacyjne(idx1(1:b),:)');
        
        klasa = predykcja > prog;
        ocena_jakosci_dac = zeros(2,2);
        for a=1:length(status_walidacyjny)
            if (klasa(a,1) == 1 && status_walidacyjny(1,a) == 1)
                    ocena_jakosci_dac(1,1) = ocena_jakosci_dac(1,1) + 1;
                elseif (klasa(a,1) == 0 && status_walidacyjny(1,a) == 0)
                    ocena_jakosci_dac(2,2) = ocena_jakosci_dac(2,2) + 1;
                elseif (klasa(a,1) == 1 && status_walidacyjny(1,a) == 0)
                    ocena_jakosci_dac(1,2) = ocena_jakosci_dac(1,2) + 1;
                else 
                    ocena_jakosci_dac(2,1) = ocena_jakosci_dac(2,1) + 1;
            end
        end
        
      dokladnosc_dac(i,b) = (ocena_jakosci_dac(1,1) + ocena_jakosci_dac(2,2)) / (ocena_jakosci_dac(1,1) + ocena_jakosci_dac(1,2) + ocena_jakosci_dac(2,1) + ocena_jakosci_dac(2,2));
      czulosc_dac(i,b) = ocena_jakosci_dac(1,1) / (ocena_jakosci_dac(1,1) + ocena_jakosci_dac(2,1));
      specyficznosc_dac(i,b) = ocena_jakosci_dac(2,2) / (ocena_jakosci_dac(2,2) + ocena_jakosci_dac(1,2));
      blad_wazony_dac(i,b) = 1 - ((czulosc_dac(i,b) + specyficznosc_dac(i,b)) / 2);

        
        
    end  
    
    
    
end
    
       
end

    





