function [ues, cells] = updateUEDropEvents(ues, cells, logFile, currentTime)
    for ueIdx = 1:length(ues)
        ue = ues(ueIdx);
        
        if ~isnan(ue.servingCell) && ue.sessionActive
            % Find serving cell
            servingCellIdx = find([cells.id] == ue.servingCell, 1);
            
            if ~isempty(servingCellIdx)
                cell = cells(servingCellIdx);
                
                dropProb = 0.001; % Base drop probability (0.1%)
                
                if cell.txPower <= cell.minTxPower + (cell.maxTxPower - cell.minTxPower)/4
                    powerPenalty = (cell.maxTxPower - (cell.minTxPower + (cell.maxTxPower - cell.minTxPower)/4) + 1) * 0.08;
                    dropProb = dropProb + powerPenalty;
                elseif cell.txPower <= cell.minTxPower + (cell.maxTxPower - cell.minTxPower)/2
                    powerPenalty = (cell.maxTxPower - (cell.minTxPower + (cell.maxTxPower - cell.minTxPower)/2) + 1) * 0.03;
                    dropProb = dropProb + powerPenalty;
                end
                
                % Signal quality factor - now more severe
                if ~isnan(ue.sinr)
                    if ue.sinr < -10
                        dropProb = dropProb + 0.15 + abs(ue.sinr + 10) * 0.02; 
                    elseif ue.sinr < -5
                        dropProb = dropProb + 0.08 + abs(ue.sinr + 5) * 0.015;
                    elseif ue.sinr < 0
                        dropProb = dropProb + 0.04;
                    elseif ue.sinr < 5
                        dropProb = dropProb + 0.01;
                    end
                end
                
                if ~isnan(ue.rsrp)
                    if ue.rsrp < -120
                        dropProb = dropProb + 0.12 + abs(ue.rsrp + 120) * 0.01;
                    elseif ue.rsrp < -115
                        dropProb = dropProb + 0.06 + abs(ue.rsrp + 115) * 0.008;
                    elseif ue.rsrp < -110
                        dropProb = dropProb + 0.03;
                    end
                end
                
                % Cell congestion factor
                if cell.cpuUsage > 95
                    dropProb = dropProb + (cell.cpuUsage - 95) * 0.015; % 1.5% per % above 95%
                elseif cell.cpuUsage > 90
                    dropProb = dropProb + (cell.cpuUsage - 90) * 0.01;
                end
                
                if cell.prbUsage > 95
                    dropProb = dropProb + (cell.prbUsage - 95) * 0.012; % 1.2% per % above 95%
                elseif cell.prbUsage > 90
                    dropProb = dropProb + (cell.prbUsage - 90) * 0.008;
                end
                
                % Traffic load factor
                loadRatio = cell.currentLoad / cell.maxCapacity;
                if loadRatio > 0.98
                    dropProb = dropProb + (loadRatio - 0.98) * 1.0; 
                elseif loadRatio > 0.95
                    dropProb = dropProb + (loadRatio - 0.95) * 0.6;
                elseif loadRatio > 0.90
                    dropProb = dropProb + (loadRatio - 0.90) * 0.2;
                end
                
                if cell.txPower <= cell.maxTxPower && (ue.rsrp < -110 || ue.sinr < -5)
                    dropProb = dropProb + 0.25; 
                end

                
                if rand() < min(0.45, dropProb)
                    cells(servingCellIdx).actualDropCount = cells(servingCellIdx).actualDropCount + 1;
                    cells(servingCellIdx).totalDropEvents = cells(servingCellIdx).totalDropEvents + 1;

                    ue.servingCell = NaN;
                    ue.rsrp = NaN;
                    ue.rsrq = NaN;
                    ue.sinr = NaN;
                    ue.sessionActive = false;
                    ue.trafficDemand = 0;
                    ue.dropCount = ue.dropCount + 1;
                    logMsg = sprintf('Drop Event: UE %d dropped from Cell %d (Total drops: %d)\n', ...
                                     ue.id, cell.id, cells(servingCellIdx).totalDropEvents);
                    fprintf(logMsg);
                    logToFile(logFile, currentTime, logMsg);
                end
            else
                ue.servingCell = NaN;
                ue.rsrp = NaN;
                ue.rsrq = NaN;
                ue.sinr = NaN;
                ue.sessionActive = false;
                ue.trafficDemand = 0;
                ue.dropCount = ue.dropCount + 1;
            end
        end
        
        ues(ueIdx) = ue;
    end
end