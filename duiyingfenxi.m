clc,clear
a=load('E:\libs\duiyingfenxiselectfeature2.txt')
T=sum(sum(a));
P=a/T;
r=sum(P,2),c=sum(P)
Row_prifile=a./repmat(sum(a,2),1,size(a,2))
B=(P-r*c)./sqrt((r*c));
[u,s,v]=svd(B,'econ')
w1=sign(repmat(sum(v),size(v,1),1))
w2=sign(repmat(sum(v),size(u,1),1))
vb=v.*w1
ub=u.*w2
lambda=diag(s).^2
ksi2square=T*(lambda)
T_ksi2square=sum(ksi2square)
con_rate=lambda/sum(lambda)
cum_rate=cumsum(con_rate)
beta=diag(r.^(-1/2))*ub;
G=beta*s
alpha=diag(c.^(-1/2))*vb;
F=alpha*s
num1=size(G,1);
rang=minmax(G(:,[1,3])');
delta=(rang(:,2)-rang(:,1)/(5*num1));
%yb=['a001';'a002';'a003';'a004';'a005';'a006';'a007';'a008';'a009';'a010';'a011';'a012';'a013';'a014';'a015';'a016';'a017';'a018';'a019';'a020';'a021';'a022';'a023';'a024';'a025';'a026';'a027';'a028';'a029';'a030';'a031';'a032';'a033';'a034';'a035';'a036';'a037';'a038';'a039';'a040';'a041';'a042';'a043';'a044';'a045';'a046';'a047';'a048';'a049';'a050';'a051';'a052';'a053';'a054';'a055';'a056';'a057';'a058';'a059';'a060';'a061';'a062';'a063';'a064';'a065';'a066';'a067';'a068';'a069';'a070';'a071';'a072';'a073';'a074';'a075';'a076';'a077';'a078';'a079';'a080';'a081';'a082';'a083';'a084';'a085';'a086';'a087';'a088';'a089';'a090';'a091';'a092';'a093';'a094';'a095';'a096';'a097';'a098';'a099';'a100';'a101';'a102';'a103';'a104'];%有8个位移就写到8，excel表8行
yb=['a001';'a002';'a003';'a004';'a005';'a006';'a007';'a008';'a009';'a010';'a011';'a012';'a013';'a014';'a015';'a016';'a017';'a018';'a019';'a020';'a021';'a022';'a023';'a024';'a025';'a026';'a027';'a028';'a029';'a030';'a031';'a032';'a033';'a034';'a035';'a036';'a037';'a038';'a039';'a040';'a041';'a042';'a043';'a044';'a045';'a046';'a047';'a048';'a049';'a050';'a051';'a052';'a053';'a054';'a055';'a056';'a057';'a058';'a059';'a060';'a061';'a062';'a063';'a064';'a065';'a066';'a067';'a068';'a069';'a070';'a071';'a072';'a073';'a074';'a075';'a076';'a077';'a078';'a079';'a080';'a081';'a082';'a083';'a084';'a085';'a086';'a087';'a088';'a089';'a090';'a091';'a092';'a093';'a094';'a095';'a096'];
%ch=['001';'002';'003';'004';'005';'006';'007';'008';'009';'010';'011';'012';'013';'014';'015';'016';'017';'018';'019';'020';'021';'022';'023';'024';'025';'026';'027';'028';'029';'030';'031';'032';'033';'034';'035';'036';'037';'038';'039';'040';'041';'042';'043';'044';'045';'046';'047';'048';'049';'050';'051';'052';'053';'054';'055';'056';'057';'058';'059';'060';'061';'062';'063';'064';'065';'066';'067';'068';'069';'070';'071';'072';'073';'074';'075';'076';'077';'078';'079';'080';'081';'082';'083';'084';'085';'086';'087';'088';'089';'090';'091';'092';'093';'094';'095';'096';'097';'098';'099';'100';'101';'102';'103';'104';'105';'106';'107';'108';'109';'110';'111';'112';'113';'114';'115';'116';'117';'118';'119';'120';'121';'122';'123';'124';'125';'126';'127';'128';'129';'130';'131';'132';'133';'134';'135';'136';'137';'138';'139';'140';'141';'142';'143';'144';'145';'146';'147';'148';'149';'150';'151';'152';'153';'154';'155';'156';'157';'158';'159';'160';'161';'162';'163';'164';'165';'166';'167';'168';'169';'170';'171';'172';'173';'174';'175';'176';'177';'178';'179';'180';'181';'182';'183';'184';'185';'186';'187';'188';'189';'190';'191';'192';'193';'194';'195';'196';'197';'198';'199';'200';'201';'202';'203';'204';'205';'206';'207';'208';'209';'210';'211';'212';'213';'214';'215';'216';'217';'218';'219';'220';'221';'222';'223'];
%ch=['001';'002';'003';'004';'005';'006';'007';'008';'009';'010';'011';'012';'013';'014';'015';'016';'017';'018';'019';'020';'021';'022';'023';'024';'025';'026';'027';'028';'029';'030';'031';'032';'033';'034';'035';'036';'037';'038';'039';'040';'041';'042';'043';'044';'045';'046';'047';'048';'049';'050';'051';'052';'053';'054';'055';'056';'057';'058';'059';'060';'061';'062';'063';'064';'065';'066';'067';'068'];
%data = [231.594;238.226;239.441;239.505;239.569;241.608;253.427;253.675;253.862;254.916;254.978;256.276;258.555;259.905;260.701;261.19;261.373;262.593;262.836;263.14;266.473;273.952;274.307;274.663;274.899;275.313;275.55;324.692;327.333;330.136;334.393;341.383;345.743;346.084;349.203;351.397;352.348;356.459;356.514;356.9;358.001;360.789;361.768;361.822;363.067;364.683;373.358;374.863;381.958;382.461;394.298;396.052;404.601;406.343;407.183;414.367;427.182;430.807;432.55;432.601;438.352;440.485;472.183;472.224;481.011;510.508;515.248;521.765]
%data=[224.292;224.686;236.993;238.194;238.242;239.441;239.489;239.537;239.585;240.494;241.592;241.64;253.427;253.66;253.706;253.893;254.916;254.962;258.571;258.617;259.307;259.905;260.685;261.19;261.373;261.739;262.562;262.836;263.11;273.967;274.322;274.677;274.899;275.343;275.564;324.692;327.303;330.166;334.378;341.369;345.757;349.203;351.397;352.362;356.487;356.528;356.57;356.9;358.015;360.803;361.781;361.822;363.04;373.384;381.946;394.286;396.028;404.572;404.615;406.343;407.183;414.353;427.169;427.52;430.781;430.82;432.55;432.588;438.339;440.46;440.497;465.084;472.183;472.214;481.031;510.519;515.269;521.745;521.775]
data=[239.441;241.608;259.905;260.701;261.19;263.14;273.952;274.663;274.899;275.55;324.692;327.333;334.393;351.397;352.348;356.514;356.57;356.9;358.001;361.768;361.822;394.298;396.052;404.601;407.183;427.182;430.807;432.55;432.601;438.352;440.485;455.402;472.183;472.224;481.011;493.348;510.508;515.248;521.765]
str_data = cellstr(num2str(data, '%.3f'));
ch= char(str_data); % 格式化为字符串
hold on
%plot(G(:,1),G(:,2),'*','Color','k','LineWidth',10)
%text(G(:,1)+0.01,G(:,2),yb)
plot(F(:,1),F(:,2),'.','Color','k','LineWidth',10)
text(F(:,1)-0.1,F(:,2),ch, 'FontSize', 10, 'FontName', 'Times New Roman')
xlabel('Dim1/arb.units','FontName','Times New Roman','Fontsize',10);
ylabel('Dim2/arb.units','FontName','Times New Roman','Fontsize',10);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);
box on;%添加边框
%xlswrite('tttttt',[diag(s),lambda,ksi2square,con_rate,cum_rate])
%ind1=find(G(:,1)>0);
%rowclass=yb(ind1)
%ind2=find(F(:,1)>0);
%colclass=ch(ind2)

