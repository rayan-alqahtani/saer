import {useState, type FormEvent} from 'react';
import type {ReactNode} from 'react';
import {APP_NAME} from './config';

export default function LoginPage({navigate,header}:{navigate:(path:string)=>void;header:ReactNode}) {
  const [name,setName]=useState('');
  const [askingName,setAskingName]=useState(false);
  const enterAsGuest=()=>setAskingName(true);
  const submitGuestName=(event:FormEvent)=>{
    event.preventDefault();
    const firstName=name.trim().split(/\s+/)[0];
    if(!firstName)return;
    sessionStorage.setItem('saer_guest','true');
    sessionStorage.setItem('saer_guest_name',firstName);
    window.dispatchEvent(new Event('saer-guest-name'));
    navigate('/');
  };

  return (
    <main className="login-screen login-screen-fit" dir="rtl" aria-label={APP_NAME}>
      {header}
      <div className="login-reference-stage">
        <img className="login-reference-image" src="/ui/login/login-final-market.png" alt={`واجهة تسجيل الدخول إلى ${APP_NAME}`} draggable={false}/>
        <button className="login-guest-only" type="button" aria-label="ادخل كزائر للموقع" onClick={enterAsGuest} />
      </div>
      {askingName && <div className="guest-name-backdrop" role="presentation" onMouseDown={event=>{if(event.target===event.currentTarget)setAskingName(false);}}>
        <form className="guest-name-dialog" role="dialog" aria-modal="true" aria-labelledby="guest-name-title" onSubmit={submitGuestName}>
          <h1 id="guest-name-title">أهلاً بك، ما اسمك الأول؟</h1>
          <label htmlFor="guest-first-name">الاسم الأول</label>
          <input id="guest-first-name" autoFocus required maxLength={40} value={name} onChange={event=>setName(event.target.value)} placeholder="اكتب اسمك الأول" />
          <button type="submit">الدخول كزائر</button>
          <button className="guest-name-cancel" type="button" onClick={()=>setAskingName(false)}>إلغاء</button>
        </form>
      </div>}
    </main>
  );
}
